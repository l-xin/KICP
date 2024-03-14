# -*- coding: utf-8 -*-

from copy import deepcopy
import torch
from torch import nn
from transformers import BertModel, BertLayer

class AdapterLayer(nn.Module):
    def __init__(self, config):
        super(AdapterLayer, self).__init__()
        bert_hidden_size = config.hidden_size
        adapter_hidden_size = config.adapter_hidden_size
        adapter_config = deepcopy(config)
        adapter_config.hidden_size = config.adapter_hidden_size
        adapter_config.intermediate_size = adapter_config.hidden_size * config.intermediate_size // config.hidden_size
        self.bert_proj = nn.Linear(bert_hidden_size, adapter_hidden_size)
        self.bert_layer = BertLayer(adapter_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.functional.gelu
        return
    
    def forward(self, bert_hidden_states, adapter_hidden_states=None, attention_mask=None):
        hidden_states = self.dropout(self.act(self.bert_proj(bert_hidden_states)))
        if adapter_hidden_states is not None:
            hidden_states = hidden_states + adapter_hidden_states
        output = self.bert_layer(
            hidden_states = hidden_states,
            attention_mask=attention_mask
        )[0]
        return output

class AdapterModel(nn.Module):
    def __init__(self, config):
        super(AdapterModel, self).__init__()
        self.adapter_layers = nn.ModuleList(AdapterLayer(config) for _ in range(config.num_hidden_layers))
        return
    
    def forward(self, bert_hidden_states, attention_mask):
        hidden_state = None
        for bert_hidden_state, adapter_layer in zip(bert_hidden_states, self.adapter_layers):
            hidden_state = adapter_layer(
                bert_hidden_states = bert_hidden_state,
                adapter_hidden_states = hidden_state,
                attention_mask = attention_mask
            )
        return hidden_state

class KModel(nn.Module):
    def __init__(self, bert_path, adapter_hidden_size, bert_tunable=False):
        super(KModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, output_hidden_states=True, add_pooling_layer=False)
        self.config = self.bert.config
        if adapter_hidden_size is None:
            adapter_hidden_size = self.config.hidden_size // 2
        self.config.adapter_hidden_size = adapter_hidden_size
        self.adapter = AdapterModel(self.config)
        self.output = nn.Linear(adapter_hidden_size, self.config.hidden_size)
        self.tune_bert(bert_tunable)
        return
    
    @classmethod
    def from_pretrained(cls, path, bert_tunable=False):
        model = torch.load(path)
        model.tune_bert(bert_tunable)
        return model
        
    def save(self, path):
        torch.save(self, path)
        return

    def tune_bert(self, mode=True):
        self.bert_tunable = mode
        if mode == True:
            for p in self.parameters():
                p.requires_grad = True
        else:
            grad_params = set([k for k, p in self.trained_parameters()])
            for k, p in self.named_parameters():
                if k in grad_params:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        return

    def trained_parameters(self):
        if self.bert_tunable:
            return self.named_parameters()
        else:
            adapter_named_parameters = [('adapter.' + k, p) for k, p in self.adapter.named_parameters()]
            output_named_parameters = [('output.' + k, p) for k, p in self.output.named_parameters()]
            return adapter_named_parameters + output_named_parameters

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        last_hidden_state = bert_outputs.last_hidden_state
        bert_hidden_states = bert_outputs.hidden_states[1:]
        if not self.bert_tunable:
            last_hidden_state = last_hidden_state.detach()
            bert_hidden_states = [hidden_state.detach() for hidden_state in bert_hidden_states]
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size()) if attention_mask is not None else None
        adapter_hidden_state = self.adapter(bert_hidden_states, extended_attention_mask)
        hidden_states = self.output(adapter_hidden_state) + last_hidden_state
        return hidden_states
