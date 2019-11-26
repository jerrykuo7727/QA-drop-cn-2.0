import re
import sys
import json
from opencc import OpenCC
from calendar import monthrange
from datetime import date, timedelta
from transformers import XLNetTokenizer
from TimeNormalizer import TimeNormalizer


# Global tools

month2int = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}

date_dur_par = ('^((\d|零|一|二|三|四|五|六|七|八|九|十)+(周年|年|岁){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(周){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(个){0,1}(月){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|号|天){0,1}){0,1}$')
date_dur_full = ('^((\d|零|一|二|三|四|五|六|七|八|九|十)+(年|岁)){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+(个){0,1}月){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+周年){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+周){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|号|天)){0,1}$')
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
date_dur_par_re = re.compile(date_dur_par)
date_dur_full_re = re.compile(date_dur_full)
date_re = re.compile(date_pattern)
dur_re = re.compile(dur_pattern)
num_par_re = re.compile('^\d+(,\d*)*(\.\d*){0,1}$')
num_full_re = re.compile('^\d+(,\d+)*(\.\d+){0,1}$')
num_match_re = re.compile('\d+(,\d+)*(\.\d+){0,1}')
num_re = re.compile('(\d+|(零|一|二|三|四|五|六|七|八|九|十|百|千|万)+)')

tn = TimeNormalizer()
tokenizer = XLNetTokenizer.from_pretrained(sys.argv[1])


# Lots of functions and functions

def get_answer_atype(A):
    if len(A['spans']) == 1:
        return A['spans'][0], 0
    elif A['number']:
        return A['number'], 1
    elif 'year' in A['date'] and len(A['date']['year']) > 0 or \
         'month' in A['date'] and len(A['date']['month']) > 0 or \
         'day' in A['date'] and len(A['date']['day']) > 0:
        # Fix bad data
        if len(A['date']['day']) > 2:
            A['date']['year'], A['date']['day'] = A['date']['day'], A['date']['year']
        answer = ''
        if A['date']['year']:
            answer += '%s年' % A['date']['year']
        if A['date']['month']:
            answer += '%d月' % month2int[A['date']['month']]
        if A['date']['day']:
            answer += '%s日' % A['date']['day']
        return answer, 2
    else: return None, -1

def find_sublist_xlnet(a, b, tokenizer):
    if not b:
        return -1
    str_b = tokenizer.convert_tokens_to_string(b)
    for i in range(len(a)-len(b)+1):
        str_a = tokenizer.convert_tokens_to_string(a[i:i+len(b)])
        if str_a == str_b:
            return i
    return -1

def parse_date_answer(answer):
    year, month, day = None, None, None
    try:
        if '年' in answer:
            year, rest = answer.split('年')
            year = int(year)
            if '月' in rest:
                month, rest = rest.split('月')
                month = int(month)
                if '日' in rest:
                    day = int(rest.split('日')[0])
                else: pass
            else: pass
        else:
            if '月' in answer:
                month, rest = answer.split('月')
                month = int(month)
                if '日' in rest:
                    day = int(rest.split('日')[0])
                else: pass
            else:
                day = int(answer.split('日')[0])
        return year, month, day
    except:
        print(answer)

def find_all(par_re, full_re, tokens):
    assert '' not in tokens
    cursor = 0
    cand_tokens = []
    found_strings = []
    for i, token in enumerate(tokens):
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens + [token]).replace(' ', '')
        match = par_re.search(cand_string)
        if not match:
            if cand_tokens:
                cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
                full_match = full_re.search(cand_string)
                if full_match:
                    found_strings.append((cursor, full_match.group()))
                cand_tokens = []
                if par_re.match(token):
                    cand_tokens.append(token)
                    cursor = i
                else:
                    cursor = i + 1
            else:
                cursor = i + 1
        else:
            cand_tokens.append(token)
    if cand_tokens:
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
        full_match = full_re.search(cand_string)
        if full_match:
            found_strings.append((cursor, full_match.group()))
    return found_strings

def find_arith_ind_op(p_tokens, answer):
    ans_num = float(answer)
    all_nums = find_all(num_par_re, num_full_re, p_tokens)
    all_nums = [(i, float(num_str.replace(',', ''))) for i, num_str in all_nums]
    for si, src_pair in enumerate(all_nums):
        src_ind, src_num = src_pair
        for ti, tgt_pair in enumerate(all_nums):
            if si == ti: continue
            tgt_ind, tgt_num = tgt_pair
            for op_type in (0,1):
                new_num = src_num + (op_type*-2+1) * tgt_num 

                # demo print
                sign = '+' if op_type == 0 else '-'
                if new_num == ans_num:
                    return src_ind, tgt_ind, op_type
    return -1, -1, -1

def find_all_date_durs(p_tokens):
    all_date_durs = find_all(date_dur_par_re, date_dur_full_re, p_tokens)
    all_dates, all_durs = [], []
    for ind, cand in all_date_durs:
        date_match = date_re.search(cand)
        if date_match and date_match.group():
            all_dates.append((ind, date_match.group()))
        dur_match = dur_re.search(cand)
        if dur_match and dur_match.group():
            all_durs.append((ind, dur_match.group()))
    return all_dates, all_durs

def find_date_dur_ind_op(p_tokens, answer):
    all_dates, all_durs = find_all_date_durs(p_tokens)
    ans_date = parse_date_answer(answer)
    
    # Parse date
    for date_ind, raw_date in all_dates:
        try:
            timestamp = json.loads(tn.parse(raw_date, timeBase='2018-12-31'))['timestamp']
            src_date = date(*map(int, timestamp.split()[0].split('-')))    
        except:
            if raw_date.endswith('年'):
                try: year = raw_date[:-1]
                except: year = chinese2int(raw_date[:-1])
                if year in range(1, 10000):
                    src_date = date(year, 1, 1)
                else: continue
            else: continue

        # Parse duration
        for dur_ind, raw_dur in all_durs:
            num_str = num_re.search(raw_dur).group()
            try: num = int(num_str)
            except: num = chinese2int(num_str)

            # date +/- duration
            for op_type in (0, 1):  # 0: plus, 1: minus
                num *= op_type * -2 + 1
                tgt_date = None
                if any(raw_dur.endswith(c) for c in ('年', '岁')):
                    tgt_date = add_years(src_date, num)
                if raw_dur.endswith('月'):
                    tgt_date = add_months(src_date, num)
                if raw_dur.endswith('周'):
                    tgt_date = add_days(src_date, num * 7)
                if any(raw_dur.endswith(c) for c in ('日', '天')):
                    tgt_date = add_days(src_date, num)
                    
                # Post-process operation result
                tgt_year, tgt_month, tgt_day = tgt_date.year, tgt_date.month, tgt_date.day
                if tgt_year == 9999: tgt_year = tgt_month = tgt_day = None
                if '年' not in raw_date: tgt_year = None
                if '月' not in raw_date: tgt_month = None
                if all(c not in raw_date for c in ('日', '号')): tgt_day = None
                tgt_date = (tgt_year, tgt_month, tgt_day)
                
                # Check answer
                correct = True
                for tgt_unit, ans_unit in zip(tgt_date, ans_date):
                    if ans_unit and tgt_unit != ans_unit:
                        correct = False
                if correct:
                    return date_ind, dur_ind, op_type
    return -1, -1, -1

def arithmetic_op(tokenizer, num_match_re, all_tokens, start_ind, end_ind, plus):
    try:
        start_cand = tokenizer.convert_tokens_to_string(all_tokens[start_ind:]).replace(' ', '')
        start_match = num_match_re.search(start_cand).group()  
        end_cand = tokenizer.convert_tokens_to_string(all_tokens[end_ind:]).replace(' ', '')
        end_match = num_match_re.search(end_cand).group()

        start_num = float(start_match)
        end_num = float(end_match)
        final_num = start_num + end_num if plus else start_num - end_num
        if final_num % 1 == 0:
            final_num = int(final_num)
        return str(final_num)
    except:
        return '1'
    
def date_duration_op(tokenizer, date_re, dur_re, tn, all_tokens, start_ind, end_ind, plus):
    try:
        start_cand = tokenizer.convert_tokens_to_string(all_tokens[start_ind:]).replace(' ', '')
        raw_date = date_re.search(start_cand).group()
        end_cand = tokenizer.convert_tokens_to_string(all_tokens[end_ind:]).replace(' ', '')
        raw_dur = dur_re.search(end_cand).group()

        # Parse date
        timestamp = json.loads(tn.parse(raw_date, timeBase='2018-12-31'))['timestamp']
        src_date = date(*map(int, timestamp.split()[0].split('-')))

        # Parse duration
        num_str = num_re.search(raw_dur).group()
        try: num = int(num_str)
        except: num = chinese2int(num_str)
        if not plus: num *= -1

        # Date +/- duration
        if any(raw_dur.endswith(c) for c in ('年', '岁')):
            tgt_date = add_years(src_date, num)
        if raw_dur.endswith('月'):
            tgt_date = add_months(src_date, num)
        if raw_dur.endswith('周'):
            tgt_date = add_days(src_date, num * 7)
        if any(raw_dur.endswith(c) for c in ('日', '天')):
            tgt_date = add_days(src_date, num)

        # Clean unexist units
        tgt_year, tgt_month, tgt_day = tgt_date.year, tgt_date.month, tgt_date.day
        if tgt_year == 9999: tgt_year = tgt_month = tgt_day = None
        if '年' not in raw_date: tgt_year = None
        if '月' not in raw_date: tgt_month = None
        if all(c not in raw_date for c in ('日', '号')): tgt_day = None

        # Formatter
        final_date = ''
        if tgt_year:
            final_date += '%d年' % tgt_year
        if tgt_month:
            final_date += '%d月' % tgt_month
        if tgt_day:
            final_date += '%d日' % tgt_day
        if final_date:
            return final_date
        else:
            return '1月'
    except:
        return '1月'

def chinese2int(chn):
    def _trans(s):
        num = 0
        if s:
            idx_q, idx_b, idx_s = s.find('千'), s.find('百'), s.find('十')
            if idx_q != -1:
                num += digit[s[idx_q - 1:idx_q]] * 1000
            if idx_b != -1:
                num += digit[s[idx_b - 1:idx_b]] * 100
            if idx_s != -1:
                num += digit.get(s[idx_s - 1:idx_s], 1) * 10
            if s[-1] in digit:
                num += digit[s[-1]]
        return num
    try:
        digit = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        chn = chn.replace('零', '')
        idx_y, idx_w = chn.rfind('亿'), chn.rfind('万')
        if idx_w < idx_y:
            idx_w = -1
        num_y, num_w = 100000000, 10000
        if idx_y != -1 and idx_w != -1:
            return trans(chn[:idx_y]) * num_y + _trans(chn[idx_y + 1:idx_w]) * num_w + _trans(chn[idx_w + 1:])
        elif idx_y != -1:
            return trans(chn[:idx_y]) * num_y + _trans(chn[idx_y + 1:])
        elif idx_w != -1:
            return _trans(chn[:idx_w]) * num_w + _trans(chn[idx_w + 1:])
        return _trans(chn)
    except: return 0

def add_years(src_date, years):
    if src_date.year + years not in range(1, 10000):
        year = 9999
    else:
        year = src_date.year + years
    month = src_date.month
    day = src_date.day
    try:
        return date(year, month, day)
    except:
        return date(year, month, day-1)  # due to 2/29

def add_months(src_date, months):
    month = src_date.month - 1 + months
    year = src_date.year + month // 12
    month = month % 12 + 1
    day = min(src_date.day, monthrange(year, month)[1])
    return date(year, month, day)

def add_days(src_date, days):
    return src_date + timedelta(days)


if __name__ == '__main__':
    for split in ('train', 'dev'):
        print('<DROP - %s set>' % split)
        data = json.load(open('dataset/drop_dataset_%s_cn_azure.json' % split))

        n_question, n_sin_span = 0, 0
        impos_arith, n_arith = 0, 0
        impos_dd, n_datedur = 0, 0

        for i, item in enumerate(data.items(), start=1):
            PID, PQA = item
            raw_passage = PQA['passage']
            p_tokens = tokenizer.tokenize(raw_passage)
            p_tokens_no_unk = p_tokens

            with open('data/%s/passage/DROP-%s' % (split, PID), 'w') as f:
                assert p_tokens == ' '.join(p_tokens).split(' ')
                f.write(' '.join(p_tokens))

            with open('data/%s/passage_no_unk/DROP-%s' % (split, PID), 'w') as f:
                assert p_tokens_no_unk == ' '.join(p_tokens_no_unk).split(' ')
                f.write(' '.join(p_tokens_no_unk))

            # QAs
            for QA in PQA['qa_pairs']:
                raw_question = QA['question']
                raw_answer, atype = get_answer_atype(QA['answer'])
                if atype < 0: continue
                q_tokens = tokenizer.tokenize(raw_question)
                q_tokens_no_unk = q_tokens
                n_question += 1

                # Single-span
                if atype == 2 and sum(c in raw_answer for c in ('年', '月', '日')) == 1:
                    a_tokens_no_unk = tokenizer.tokenize(raw_answer[:-1])
                    if a_tokens_no_unk and a_tokens_no_unk[0] == '▁':
                        a_tokens_no_unk = a_tokens_no_unk[1:]
                    answer_start = find_sublist_xlnet(p_tokens_no_unk, a_tokens_no_unk, tokenizer)
                else:
                    a_tokens_no_unk = tokenizer.tokenize(raw_answer)
                    if a_tokens_no_unk and a_tokens_no_unk[0] == '▁':
                        a_tokens_no_unk = a_tokens_no_unk[1:]
                    answer_start = find_sublist_xlnet(p_tokens_no_unk, a_tokens_no_unk, tokenizer)
                if answer_start >= 0:
                    answer_end = answer_start + len(a_tokens_no_unk) - 1
                    op_type = 0
                    n_sin_span += 1

                # Arithmetic
                if atype == 1 and answer_start < 0:
                    impos_arith += 1
                    answer_start, answer_end, sign_type = find_arith_ind_op(p_tokens_no_unk, raw_answer)
                    op_type = 1 + sign_type
                    if answer_start >= 0:
                        n_arith += 1
                    else: continue

                # Date-duration
                if atype == 2 and answer_start < 0:
                    impos_dd += 1
                    answer_start, answer_end, sign_type = find_date_dur_ind_op(p_tokens_no_unk, raw_answer)
                    op_type = 3 + sign_type
                    if answer_start >= 0:
                        n_datedur += 1
                    else: continue

                # Save processed data
                if answer_start < 0: continue
                QID = QA['query_id']
                with open('data/%s/question/DROP-%s|%s' % (split, PID, QID), 'w') as f:
                    assert q_tokens  == ' '.join(q_tokens).split(' ')
                    f.write(' '.join(q_tokens))
                with open('data/%s/question_no_unk/DROP-%s|%s' % (split, PID, QID), 'w') as f:
                    assert q_tokens_no_unk == ' '.join(q_tokens_no_unk).split(' ')
                    f.write(' '.join(q_tokens_no_unk))
                with open('data/%s/answer/DROP-%s|%s' % (split, PID, QID), 'w') as f:
                    f.write('%s' % raw_answer)
                with open('data/%s/span/DROP-%s|%s' % (split, PID, QID), 'w') as f:
                    f.write('%d %d %d' % (op_type, answer_start, answer_end))

            print('passage: %d/%d\r' % (i, len(data)), end='')
        print('\npos_sin, all_que: %d/%d' % (n_sin_span, n_question))
        print('pos_ari, imp_ari: %d/%d' % (n_arith, impos_arith))
        print('pos_ddu, imp_ddu: %d/%d' % (n_datedur, impos_dd))
