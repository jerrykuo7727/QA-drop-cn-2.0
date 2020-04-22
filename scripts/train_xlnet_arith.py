import re
import os
import sys
import json
import numpy as np
from os.path import join
from copy import deepcopy
from TimeNormalizer import TimeNormalizer
from prepare_xlnet_drop_data import arithmetic_op
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

import torch
from utils import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLNetTokenizer, XLNetForQuestionAnswering

np.random.seed(42)
torch.manual_seed(42)

# Global tools
norm_tokenizer = BertTokenizer.from_pretrained('/home/M10815022/Models/bert-wwm-ext')
tokenizer = XLNetTokenizer.from_pretrained(sys.argv[2])

num_par_re = re.compile('^\d+(,\d*)*(\.\d*){0,1}$')
num_full_re = re.compile('^\d+(,\d+)*(\.\d+){0,1}$')
num_match_re = re.compile('\d+(,\d+)*(\.\d+){0,1}')


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
        
        with open('data/%s/span/%s' % (self.split, question_id)) as f:
            span = f.read().split(' ')
            op_type = int(span[0])
            answer_start = int(span[1])
            answer_end = int(span[2])

        # Truncate length to 512
        diff = len(question) + len(passage) - 511
        if self.split == 'train':
            if diff > 0:
                if answer_start > 510 - diff or answer_end > 510 - diff:
                    passage = passage[diff:]
                    passage_no_unk = passage_no_unk[diff:]
                    answer_start -= diff
                    answer_end -= diff
                else:
                    passage = passage[:-diff]
                    passage_no_unk = passage_no_unk[:-diff]
            if answer_start not in range(0,512) or answer_end not in range(0,512) or \
               answer_start >= len(passage) or answer_end >= len(passage) or op_type not in range(4):
                answer_start = answer_end = op_type = -100
        else:
            if self.bwd:
                passage = passage[diff:]
                passage_no_unk = passage_no_unk[diff:]
            else:
                passage = passage[:-diff]
                passage_no_unk = passage_no_unk[:-diff]
        

        passage.append(self.tokenizer.sep_token)
        passage_no_unk.append(self.tokenizer.sep_token)
        input_tokens = passage + question
        input_tokens_no_unk = passage_no_unk + question_no_unk

        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        attention_mask = torch.FloatTensor([1 for _ in input_tokens])
        token_type_ids = torch.LongTensor([0 for _ in passage] + [1 for _ in question])
        if self.split == 'train':
            start_positions = torch.LongTensor([answer_start]).squeeze(0)
            end_positions = torch.LongTensor([answer_end]).squeeze(0)
            return input_ids, attention_mask, token_type_ids, start_positions, end_positions
        else:
            return input_ids, attention_mask, token_type_ids, op_type, input_tokens_no_unk, answer
    
def get_dataloader(split, tokenizer, bwd=False, batch_size=1, num_workers=0, prefix=None):
    def train_collate_fn(batch):
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        start_positions = torch.stack(start_positions)
        end_positions = torch.stack(end_positions)
        return input_ids, attention_mask, token_type_ids, start_positions, end_positions
    
    def test_collate_fn(batch):
        input_ids, attention_mask, token_type_ids, op_types, input_tokens_no_unk, answer = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        return input_ids, attention_mask, token_type_ids, op_types, input_tokens_no_unk, answer
    
    shuffle = (split == 'train')
    collate_fn = train_collate_fn if split == 'train' else test_collate_fn
    dataset = XLNetDataset(split, tokenizer, bwd, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader


def validate_dataset(model, split, tokenizer, topk=5):
    assert split in ('dev', 'test')
    dataloader = get_dataloader(split, tokenizer, bwd=False, batch_size=8, num_workers=8)
    em, f1, count = 0, 0, 0
    
    model.start_n_top = topk
    model.end_n_top = topk
    model.eval()
    for bi, batch in enumerate(dataloader, start=1):
        batch = (*(tensor.cuda(device) for tensor in batch[:3]), *batch[3:])
        input_ids, attention_mask, token_type_ids, op_types, input_tokens_no_unk, answers = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        start_index = outputs[1]
        end_index = outputs[3].view(-1, model.end_n_top, model.start_n_top).permute([0,2,1])[:,:,0]
        for i, answer in enumerate(answers):
            preds = []
            for k in range(model.start_n_top):
                op_type = op_types[i]
                assert op_type in (0,1,2,3)
                start_ind = start_index[i][k]
                end_ind = end_index[i][k]
                input_tokens = input_tokens_no_unk[i]
                pred = arithmetic_op(tokenizer, num_par_re, num_full_re, num_match_re, \
                                     input_tokens, start_ind, end_ind, op=op_type)
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
    model = XLNetForQuestionAnswering.from_pretrained(model_path)
    ckpt = torch.load('models/finetune.ckpt')
    model.load_state_dict(ckpt)
    device = torch.device(sys.argv[1])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('train', tokenizer, batch_size=batch_size, num_workers=8)

    print('Start training...')
    while True:
        for batch in dataloader:
            if batch[3].min().item() == -100: 
                continue
            batch = (tensor.cuda(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
            model.train()
            loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                         start_positions=start_positions, end_positions=end_positions)[0]
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

            if patience > 5 or step >= 200000:
                print('Finish training. Scoring 1-5 best results...')
                save_path = join(sys.argv[3], 'finetune.ckpt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                for k in range(1, 6):
                    validate(model, tokenizer, topk=k)
                del model, dataloader
                exit(0)
