import os
import sys
import json
import numpy as np
from os.path import join
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score

import torch
from utils import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetForSequenceClassification

np.random.seed(42)
torch.manual_seed(42)


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
        
        with open('data/%s/span/%s' % (self.split, question_id)) as f:
            span = f.read().split(' ')
            op_type = int(span[0])
        
            
        # Truncate length to 512
        diff = len(question) + len(passage) - 511
        if diff > 0:
            passage = passage[:-diff]
            passage_no_unk = passage_no_unk[:-diff]

        passage.append(self.tokenizer.sep_token)
        passage_no_unk.append(self.tokenizer.sep_token)
        input_tokens = passage + question
        input_tokens_no_unk = passage_no_unk + question_no_unk

        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        attention_mask = torch.FloatTensor([1 for _ in input_tokens])
        token_type_ids = torch.LongTensor([0 for _ in passage] + [1 for _ in question])
        op_types = torch.LongTensor([op_type]).squeeze(0)
        return input_ids, attention_mask, token_type_ids, op_types
    
def get_dataloader(split, tokenizer, bwd=False, batch_size=1, num_workers=0, prefix=None):
    def collate_fn(batch):
        input_ids, attention_mask, token_type_ids, op_types = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        op_types = torch.stack(op_types)
        return input_ids, attention_mask, token_type_ids, op_types
    
    shuffle = split == 'train'
    dataset = XLNetDataset(split, tokenizer, bwd, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader


def validate_dataset(model, split, tokenizer):
    assert split == 'dev'
    model.eval()
    y_true, y_pred = np.empty([0]), np.empty([0])
    
    dataloader = get_dataloader(split, tokenizer, batch_size=32, num_workers=8)
    n_batch = len(dataloader)
    for i, batch in enumerate(dataloader, start=1):
        batch = (tensor.cuda(device) for tensor in batch)
        input_ids, attention_mask, token_type_ids, op_types = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=op_types)
        y_true = np.hstack([y_true, op_types.cpu().numpy()])
        y_pred = np.hstack([y_pred, outputs[1].argmax(-1).cpu().numpy()])
        print('%d/%d\r' % (i, n_batch), end='')
    del dataloader
    micro_f1 = f1_score(y_true, y_pred, labels=[0,1,2,3,4], average='micro')
    accuracies = []
    for n in range(5):
        ind = np.where(y_true == n)
        accuracies.append(accuracy_score(y_true[ind], y_pred[ind]))
    return micro_f1, accuracies

def validate(model, tokenizer):
    val_micro_f1, val_accuracies = validate_dataset(model, 'dev', tokenizer)
    print(' (dev) | micro_f1=%.3f | 0_acc=%.3f, 1_acc=%.3f, 2_acc=%.3f, 3_acc=%.3f, 4_acc=%.3f' \
        % (100*val_micro_f1, *(100*acc for acc in val_accuracies)))
    return val_micro_f1


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
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    model = XLNetForSequenceClassification.from_pretrained(model_path)

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
            batch = (tensor.cuda(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, op_types = batch
            model.train()
            loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=op_types)[0]
            loss.backward()

            step += 1
            print('step %d | Training...\r' % step, end='')
            if step % update_stepsize == 0:
                optimizer.step()
                optimizer.zero_grad()
    
            if step % 3000 == 0:
                print("step %d | Validating..." % step)
                val_f1 = validate(model, tokenizer)
                if val_f1 > best_val:
                    patience = 0
                    best_val = val_f1
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    patience += 1

            if patience > 20 or step >= 200000:
                print('Finish training.')
                save_path = join(sys.argv[3], 'op_finetune.ckpt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                del model, dataloader
                exit(0)
