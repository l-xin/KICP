# -*- coding: utf-8 -*-

import logging
import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from mwp_decoder import Decoder

class MLM(nn.Module):
    def __init__(self, lm):
        super(MLM, self).__init__()
        self.lm = lm
        self.config = lm.config
        self.cls = BertOnlyMLMHead(self.config)
        self.loss_func = nn.CrossEntropyLoss()
        return
    
    @classmethod
    def from_pretrained(cls, path):
        model = torch.load(path)
        return model
        
    def save(self, path):
        torch.save(self, path)
        return
    
    def init_cls(self, bert_path):
        bert = BertForMaskedLM.from_pretrained(bert_path)
        self.cls.load_state_dict(bert.cls.state_dict())
        return
    
    def trained_parameters(self):
        lm_trained_parameters = [('lm.' + k, p) for k, p in self.lm.trained_parameters()]
        cls_named_parameters = [('cls.' + k, p) for k, p in self.cls.named_parameters()]
        return lm_trained_parameters + cls_named_parameters
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        hidden_states = self.lm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        scores = self.cls(hidden_states)
        mlm_loss = None
        if labels is not None:
            nan_mask = torch.isnan(scores).sum(dim=-1) > 0
            if nan_mask.sum().item() > 0:
                batch_index = torch.arange(nan_mask.size(0), device=nan_mask.device)
                ignored_index = batch_index[nan_mask.sum(dim=-1) > 0].cpu().tolist()
                logging.warning(f"ignore batch index {ignored_index} for nan")
                scores.masked_fill_(nan_mask.unsqueeze(-1), 0)
                labels.masked_fill_(nan_mask, -100)
            mlm_loss = self.loss_func(scores.view(-1, self.config.vocab_size), labels.view(-1))
        return mlm_loss, scores

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        return

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class QAModel(nn.Module):
    def __init__(self, lm, single_label=False, pos_weight=5):
        super(QAModel, self).__init__()
        self.lm = lm
        hidden_size = lm.config.hidden_size
        self.output = nn.Linear(hidden_size, 1)
        
        self.pool = BertPooler(lm.config)
        self.dropout = nn.Dropout(lm.config.hidden_dropout_prob)
        self.single_label = single_label
        self.pos_weight = pos_weight
        return
    
    def trained_parameters(self):
        named_parameters = [('lm.' + k, p) for k, p in self.lm.trained_parameters()]
        named_parameters += [('output.' + k, p) for k, p in self.output.named_parameters()]
        named_parameters += [('pool.' + k, p) for k, p in self.pool.named_parameters()]
        named_parameters += [('dropout.' + k, p) for k, p in self.dropout.named_parameters()]
        return named_parameters

    def forward(self, ent_candidates, prop_candidates, type_map, pos_map, labels=None):
        # candidate = "question" + "candidate_answer (a property or an entity)"
        # To process candidates with LM in one go => Gather candidates for all questions into a list (instead of ques-cand matrix)
        # To faster computation => Separate candidate list into ent_candidates & prop_candidates, reduce padding since property may be much longer than entity (e.g., very long text)
        # pos_map & type_map => Indicate position of each candidate for each question
        # type_map: n_question, type_map[i] = i-th question's candidates in which list, 0 == ent_candidates, 1 == prop_candidates
        # pos_map: n_question * n_max_candidate, pos_map[i, j] = i-th question's j-th candidate index in ent/prop_candidate list, -1 == pad
        # all above processed in QADataset @ dataset.py

        if ent_candidates is not None:
            ent_cand_hidden = self.lm(ent_candidates.input_ids, ent_candidates.attention_mask, ent_candidates.token_type_ids)[:,0,:]
        if prop_candidates is not None:
            prop_cand_hidden = self.lm(prop_candidates.input_ids, prop_candidates.attention_mask, prop_candidates.token_type_ids)[:,0,:]
        # recover candidate list into question-candidate matrix, with pos_map & type_map
        if ent_candidates is None or prop_candidates is None:
            if ent_candidates is not None:
                cand_hidden = ent_cand_hidden
            else:
                cand_hidden = prop_cand_hidden
            score_hidden = cand_hidden[pos_map]
        else:
            # stack ent_cand_hidden & prop_cand_hidden to vectorize recover operation
            # with different size(0) => pad the short cand_hidden
            ent_size = ent_cand_hidden.size(0)
            prop_size = prop_cand_hidden.size(0)
            if ent_size != prop_size:
                max_size = max(ent_size, prop_size)
                dim_size = ent_cand_hidden.size(1)
                device = ent_cand_hidden.device
                if ent_size < max_size:
                    zero_size = max_size - ent_size
                    ent_cand_hidden = torch.cat((ent_cand_hidden, torch.zeros(zero_size, dim_size, device=device)), dim=0)
                if prop_size < max_size:
                    zero_size = max_size - prop_size
                    prop_cand_hidden = torch.cat((prop_cand_hidden, torch.zeros(zero_size, dim_size, device=device)), dim=0)
            cand_hidden = torch.stack([ent_cand_hidden, prop_cand_hidden], dim=0)
            type_map = type_map.unsqueeze(-1).expand(-1, pos_map.size(1))
            score_hidden = cand_hidden[type_map, pos_map]
        score = self.output(self.dropout(self.pool(score_hidden))).squeeze(-1)
        if self.single_label:
            score.masked_fill_(pos_map == -1, -float('inf'))
        else:
            score.masked_fill_(pos_map == -1, 0)
        if labels is not None:
            if self.single_label:
                loss = torch.nn.functional.cross_entropy(score, labels, reduction="mean")
            else:
                weight = (pos_map != -1).float()
                pos_weight = torch.ones(weight.size(-1), device=weight.device) * self.pos_weight
                loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels.float(), weight, pos_weight=pos_weight, reduction="sum") / weight.sum()
        else:
            loss = None
        return score, loss

class MWPModel(nn.Module):
    def __init__(self, lm, class_list, op_set, class_pad_idx, max_decode_length):
        super(MWPModel, self).__init__()
        self.class_pad_idx = class_pad_idx
        self.lm = lm
        self.pool = BertPooler(lm.config)
        hidden_size = lm.config.hidden_size
        dropout = lm.config.hidden_dropout_prob
        self.output = Decoder(class_list, op_set, hidden_size, max_decode_length, dropout)
        return
    
    def trained_parameters(self):
        named_parameters = [('lm.' + k, p) for k, p in self.lm.trained_parameters()]
        named_parameters += [('output.' + k, p) for k, p in self.output.named_parameters()]
        named_parameters += [('pool.' + k, p) for k, p in self.pool.named_parameters()]
        return named_parameters
    
    def forward(self, questions, num_pos, labels=None):
        encoder_masks = questions.attention_mask
        encoder_outputs = self.lm(questions.input_ids, questions.attention_mask, questions.token_type_ids)
        encoder_hidden = self.pool(encoder_outputs[:, 0, :])
        outputs, seq = self.output(encoder_hidden, encoder_outputs, encoder_masks, num_pos, labels)
        outputs = torch.stack(outputs, dim=1)
        seq = torch.cat(seq, dim=1)
        if labels is not None:
            label_mask = labels != self.class_pad_idx
            # decoder output log_softmax score
            loss = torch.nn.functional.nll_loss(outputs[label_mask], labels[label_mask], reduction="mean")
        else:
            loss = None
        return seq, loss
