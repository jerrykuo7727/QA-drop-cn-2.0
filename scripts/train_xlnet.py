import re
import os
import sys
import json
import numpy as np
from os.path import join
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from TimeNormalizer import TimeNormalizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLNetTokenizer
from prepare_xlnet_drop_data import arithmetic_op, date_duration_op
from transformers.modeling_xlnet import XLNetPreTrainedModel, XLNetModel
from transformers.modeling_utils import PoolerStartLogits, PoolerEndLogits

from utils import AdamW
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

np.random.seed(42)
torch.manual_seed(42)


# Global tools
norm_tokenizer = BertTokenizer.from_pretrained('/home/M10815022/Models/bert-wwm-ext')
tokenizer = XLNetTokenizer.from_pretrained(sys.argv[2])

date_pattern = ('((\d|零|一|二|三|四|五|六|七|八|九|十)+年'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+月){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|号)){0,1}|'
                '(\d|零|一|二|三|四|五|六|七|八|九|十)+月'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|号)){0,1}|'
                '(\d|零|一|二|三|四|五|六|七|八|九|十)+(日|号))')
dur_pattern = ('((\d|零|一|二|三|四|五|六|七|八|九|十)+(周年|年|岁)'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+个月'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+周'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+(日|天))')
date_re = re.compile(date_pattern)
dur_re = re.compile(dur_pattern)
num_match_re = re.compile('\d+(,\d+)*(\.\d+){0,1}')
tn = TimeNormalizer()


# QA model
class PoolerAnswerMode(nn.Module):
    def __init__(self, config, num_mode):
        super(PoolerAnswerMode, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, num_mode)

    def forward(self, hidden_states, start_states=None, start_positions=None, 
                end_states=None, end_positions=None, cls_index=None):
        hsz = hidden_states.shape[-1]
        assert start_positions is not None and end_positions is not None or \
               start_states is not None and end_states is not None
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
            
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
            end_positions = end_positions[:, None, None].expand(-1, -1, hsz)
            end_states = hidden_states.gather(-2, end_positions).squeeze(-2)
        else:
            sntop = end_states.shape[-1]
            cls_token_state = cls_token_state.unsqueeze(1).expand(-1, sntop, -1)
            start_states = start_states.unsqueeze(1).expand(-1, sntop, -1)
            end_states = end_states.permute([0,2,1])

        x = self.dense_0(torch.cat([cls_token_state, start_states, end_states], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x

class XLNetCalculator(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetCalculator, self).__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerMode(config, num_mode=5)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, cls_index=None, 
                start_positions=None, end_positions=None, op_types=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states)
        outputs = transformer_outputs[1:]

        if start_positions is not None and end_positions is not None and op_types is not None:
            for x in (cls_index, start_positions, end_positions, op_types):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            end_logits = self.end_logits(hidden_states, start_positions=start_positions)
            cls_logits = self.answer_class(hidden_states, start_positions=start_positions, \
                                           end_positions=end_positions, cls_index=cls_index)
            
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            cls_loss = loss_fct(cls_logits, op_types)
            total_loss = start_loss + end_loss
            outputs = (total_loss,) + outputs
        else:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)

            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states)
            end_log_probs = F.softmax(end_logits, dim=1)

            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            end_states = torch.einsum("blh,bls->bhs", hidden_states, end_log_probs)
            cls_logits = self.answer_class(hidden_states, cls_index=cls_index, \
                                           start_states=start_states, end_states=end_states)
            
            end_top_log_probs = end_top_log_probs.view(-1, self.end_n_top, self.start_n_top).permute([0,2,1])
            end_top_index = end_top_index.view(-1, self.end_n_top, self.start_n_top).permute([0,2,1])
            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs
        return outputs
    
    
# Dataloader
class XLNetDataset(Dataset):
    def __init__(self, split, tokenizer, bwd=False, prefix=None):
        assert split in ('train', 'dev', 'test')
        self.split = split
        self.question_list = os.listdir('data/%s/question' % split)
        self.tokenizer = tokenizer
        self.bwd = bwd
        if prefix:
             self.question_list = [q for q in self.question_list if q.startswith(prefix)]

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, i):
        question_id = self.question_list[i]
        with open('data/%s/passage/%s' % (self.split, question_id.split('|')[0])) as f:
            passage = f.read().split(' ')
        with open('data/%s/passage_no_unk/%s' % (self.split, question_id.split('|')[0])) as f:
            passage_no_unk = f.read().split(' ')

        with open('data/%s/question/%s' % (self.split, question_id)) as f:
            question = f.read().split(' ')
            question.append(self.tokenizer.sep_token)
            question.append(self.tokenizer.cls_token)
        with open('data/%s/question_no_unk/%s' % (self.split, question_id)) as f:
            question_no_unk = f.read().split(' ')
            question_no_unk.append(self.tokenizer.sep_token)
            question_no_unk.append(self.tokenizer.cls_token)

        with open('data/%s/answer/%s' % (self.split, question_id)) as f:
            answer = [line.strip() for line in f]
        
        if self.split == 'train':
            with open('data/%s/span/%s' % (self.split, question_id)) as f:
                span = f.read().split(' ')
                op_type = int(span[0])
                answer_start = int(span[1])
                answer_end = int(span[2])
        else:
            op_type = answer_start = answer_end = -100
        
            
        # Truncate length to 512
        diff = len(question) + len(passage) - 511
        if diff > 0:
            if self.split == 'train':
                if answer_start > 510 - diff or answer_end > 510 - diff:
                    passage = passage[diff:]
                    passage_no_unk = passage_no_unk[diff:]
                    answer_start -= diff
                    answer_end -= diff
                else:
                    passage = passage[:-diff]
                    passage_no_unk = passage_no_unk[:-diff]
            else:
                if self.bwd:
                    passage = passage[diff:]
                    passage_no_unk = passage_no_unk[diff:]
                else:
                    passage = passage[:-diff]
                    passage_no_unk = passage_no_unk[:-diff]
        if answer_start not in range(0,512) or answer_end not in range(0,512) or \
           answer_start >= len(passage) or answer_end >= len(passage) or op_type not in range(0,5):
            answer_start = answer_end = op_type = -100

        passage.append(self.tokenizer.sep_token)
        passage_no_unk.append(self.tokenizer.sep_token)
        input_tokens = passage + question
        input_tokens_no_unk = passage_no_unk + question_no_unk

        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        attention_mask = torch.FloatTensor([1 for _ in input_tokens])
        token_type_ids = torch.LongTensor([0 for _ in passage] + [1 for _ in question])
        cls_index = torch.LongTensor([len(input_tokens) - 1]).squeeze(0)
        if self.split == 'train':
            start_positions = torch.LongTensor([answer_start]).squeeze(0)
            end_positions = torch.LongTensor([answer_end]).squeeze(0)
            op_types = torch.LongTensor([op_type]).squeeze(0)
            return input_ids, attention_mask, token_type_ids, cls_index, start_positions, end_positions, op_types
        else:
            return input_ids, attention_mask, token_type_ids, cls_index, input_tokens_no_unk, answer

def get_dataloader(model_type, split, tokenizer, bwd=False, batch_size=1, num_workers=0, prefix=None):
    
    def train_collate_fn(batch):
        input_ids, attention_mask, token_type_ids, cls_index, start_positions, end_positions, op_types = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        cls_index = torch.stack(cls_index)
        start_positions = torch.stack(start_positions)
        end_positions = torch.stack(end_positions)
        op_types = torch.stack(op_types)
        return input_ids, attention_mask, token_type_ids, cls_index, start_positions, end_positions, op_types
    
    def test_collate_fn(batch):
        input_ids, attention_mask, token_type_ids, cls_index, input_tokens_no_unk, answers = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        cls_index = torch.stack(cls_index)
        return input_ids, attention_mask, token_type_ids, cls_index, input_tokens_no_unk, answers
    
    assert model_type in ('bert', 'xlnet')
    shuffle = False #split == 'train'
    collate_fn = train_collate_fn if split == 'train' else test_collate_fn
    if model_type == 'bert':
        dataset = BertDataset(split, tokenizer, bwd, prefix)
    elif model_type == 'xlnet':
        dataset = XLNetDataset(split, tokenizer, bwd, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader



def validate_dataset(model, split, tokenizer, topk=5):
    assert split in ('dev', 'test')
    dataloader = get_dataloader('xlnet', split, tokenizer, bwd=False, \
                        batch_size=8, num_workers=8)
    em, f1, count = 0, 0, 0
    
    model.start_n_top = topk
    model.end_n_top = topk
    model.eval()
    for batch in dataloader:
        batch = (*(tensor.cuda(device) for tensor in batch[:-2]), *batch[-2:])
        input_ids, attention_mask, token_type_ids, cls_index, input_tokens_no_unk, answers = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, cls_index=cls_index)

        start_index = outputs[1]
        end_index = outputs[3][:,:,0]
        op_types = outputs[4]
        for i, answer in enumerate(answers):
            preds = []
            for k in range(model.start_n_top):
                op_type = op_types[i][k].argmax().item()
                if op_type == 0:
                    pred_tokens = input_tokens_no_unk[i][start_index[i][k]:end_index[i][k] + 1]
                    pred = tokenizer.convert_tokens_to_string(pred_tokens)
                elif op_type == 1:
                    pred = arithmetic_op(tokenizer, num_match_re, input_tokens_no_unk[i], start_index[i][k], end_index[i][k], plus=True)
                elif op_type == 2:
                    pred = arithmetic_op(tokenizer, num_match_re, input_tokens_no_unk[i], start_index[i][k], end_index[i][k], plus=False)
                elif op_type == 3:
                    pred = date_duration_op(tokenizer, date_re, dur_re, tn, input_tokens_no_unk[i], start_index[i][k], end_index[i][k], plus=True)
                elif op_type == 4:
                    pred = date_duration_op(tokenizer, date_re, dur_re, tn, input_tokens_no_unk[i], start_index[i][k], end_index[i][k], plus=False)
                preds.append(pred)

            norm_preds_tokens = [norm_tokenizer.basic_tokenizer.tokenize(pred) for pred in preds]
            norm_preds = [norm_tokenizer.convert_tokens_to_string(norm_pred_tokens) for norm_pred_tokens in norm_preds_tokens]
            norm_answer_tokens = [norm_tokenizer.basic_tokenizer.tokenize(ans) for ans in answer]
            norm_answer = [norm_tokenizer.convert_tokens_to_string(ans_tokens) for ans_tokens in norm_answer_tokens]

            em += max(metric_max_over_ground_truths(exact_match_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            f1 += max(metric_max_over_ground_truths(f1_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            count += 1
    del dataloader
    return em, f1, count

def validate(model, tokenizer, topk=1):
    # Valid set
    val_em, val_f1, val_count = validate_dataset(model, 'dev', tokenizer, topk)
    val_avg_em = 100 * val_em / val_count
    val_avg_f1 = 100 * val_f1 / val_count

    # Test set
    test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, topk)
    test_avg_em = 100 * test_em / test_count
    test_avg_f1 = 100 * test_f1 / test_count
    
    print('%d-best | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
        % (topk, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))
    return val_avg_f1


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print('Usage: python3 train_xlnet.py cuda:<n> <model_path> <save_path>')
        exit(1)


    # Config
    lr = 2e-5
    batch_size = 4
    accumulate_batch_size = 32
    
    assert accumulate_batch_size % batch_size == 0
    update_stepsize = accumulate_batch_size // batch_size


    dataset = sys.argv[4:]
    model_path = sys.argv[2]
    model = XLNetCalculator.from_pretrained(model_path)

    device = torch.device(sys.argv[1])
    model.to(device)
    model.start_n_top = 5
    model.end_n_top = 5

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('xlnet', 'train', tokenizer, batch_size=batch_size, num_workers=0)

    print('Start training...')
    while True:
        for batch in dataloader:
            batch = (tensor.cuda(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, cls_index, start_positions, end_positions, op_types = batch
            model.train()
            if op_types.max() >= 0:
                loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, cls_index=cls_index,
                             start_positions=start_positions, end_positions=end_positions, op_types=op_types)[0]
                loss.backward()

            step += 1
            print('step %d | Training...\r' % step, end='')
            if step % update_stepsize == 0:
                optimizer.step()
                optimizer.zero_grad()
    
            if step % 3000 == 0:
                print("step %d | Validating..." % step)
                val_f1 = validate(model, tokenizer, topk=5)
                if val_f1 > best_val:
                    patience = 0
                    best_val = val_f1
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    patience += 1

            if patience > 20 or step >= 200000:
                print('Finish training. Scoring 1-5 best results...')
                save_path = join(sys.argv[3], 'finetune.ckpt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                for k in range(1, 6):
                    validate(model, tokenizer, topk=k)
                del model, dataloader
                exit(0)
