# -*- coding: utf-8 -*-

import torch

def get_sum_f1(score, label):
    f1 = torch.zeros(score.size(0)).to(score.device)
    tp = ((score == label) & score).int().sum(dim=-1)
    mask = tp != 0
    p = tp[mask] / score[mask].sum(dim=-1)
    r = tp[mask] / label[mask].sum(dim=-1)
    f1[mask] = (2 * p * r) / (p + r)
    f1 = f1.sum().item()
    return f1

def get_sum_em(score, label):
    return ((score == label).sum(dim=-1) == score.size(-1)).sum().item()

def pre_solver(pre_equ):
    op_list = set(['+', '-', '/', '*', '^'])
    status = True
    stack = []
    for elem in pre_equ:
        if elem in op_list:
            stack.append((elem, False))
        else:
            if type(elem) is str and '%' in elem:
                elem = float(elem[:-1]) / 100.0
            else:
                elem = float(elem)
            while len(stack) >= 2 and stack[-1][1]:
                opnd = stack.pop()[0]
                op = stack.pop()[0]
                if op == "+":
                    elem = opnd + elem
                elif op == "-":
                    elem = opnd - elem
                elif op == "*":
                    elem = opnd * elem
                elif op == "/":
                    elem = opnd / elem
                elif op == "^":
                    elem = opnd ** elem
                else:
                    status = False
                    break
            if status:
                stack.append((elem, True))
            else:
                break
    if status and len(stack) == 1 and stack[0][1]:
        answer = stack.pop()[0]
    else:
        answer = None
    return answer

def compute_answer(seq_idx_list, class_list, num_list, class_pad_idx):
    seq_tokens = []
    for idx in seq_idx_list:
        if idx != class_pad_idx:
            seq_tokens.append(class_list[idx])
        else:
            break
    seq_equ = []
    for token in seq_tokens:
        if "temp_" in token:
            seq_equ.append(num_list[ord(token[-1]) - ord('a')])
        else:
            seq_equ.append(token)
    try:
        answer = pre_solver(seq_equ)
    except:
        answer = None
    return answer

def get_sum_mwp_acc(seqs, answers, class_list, num_list, class_pad_idx):
    sum_acc = 0
    seqs = seqs.cpu().tolist()
    for data_idx, seq in enumerate(seqs):
        pred_ans = compute_answer(seq, class_list, num_list[data_idx], class_pad_idx)
        if pred_ans is not None and abs(pred_ans - answers[data_idx]) < 1e-5:
            sum_acc += 1
    return sum_acc
