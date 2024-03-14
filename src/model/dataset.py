# -*- coding: utf-8 -*-

import logging
import json
import os
import pickle
import random
import torch
from torch.utils import data
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification

def load_qa_dataset(path):
    dataset = []
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)
    return dataset

class KG2CorpusDataset(data.Dataset):
    def __init__(self, tokenizer, cache_root, lesson=1, max_prop_len=256, max_name_len=32, mlm_probability=0.15, kg_ent_path=None, kg_prop_path=None, ent_path=None, rel_path=None, prop_path=None, lang="zh"):
        super(KG2CorpusDataset, self).__init__()
        self.cache_root = cache_root
        self.sep = ' '                      # for recognizing word segmentation
        self.lang = lang                    # for controlling property/entity/relation string length
        self.period = tokenizer.sep_token   # for joining evidence and conclusion
        self.max_prop_len = max_prop_len    # max length of property string
        self.max_name_len = max_name_len    # max length of entity & relation name string
        self.mlm_probability = mlm_probability
        self.replace_probability = 0.8      # 0.8
        self.random_probability = 0.5       # 0.2 * 0.5
        self.step_2_probability = 0.5       # probability of 3-triple sentence, otherwise 2-triple sentence (when 3-triple are available)
        self.tokenizer = tokenizer          # TokenizerFast
        self.bert_collator = DataCollatorForTokenClassification(tokenizer, padding="longest", label_pad_token_id=-100, return_tensors="pt")
        self.bert_max_len = 512
        self.load_kg(kg_ent_path, kg_prop_path, ent_path, rel_path, prop_path)
        self.set_lesson(lesson)
        return

    def __len__(self):
        if self.lesson == 1:
            size = len(self.kg_ent_list) + len(self.kg_prop_list)
        elif self.lesson == 2:
            size = len(self.mh_kg_list) + len(self.mo_kg_list)
        elif self.lesson == 3:
            size = len(self.mh_kg_list) + len(self.mo_kg_list)
        else:
            size = 0
        return size
    
    def __getitem__(self, index):
        if self.lesson == 1:
            data = self.sample_lesson_1(index)
        elif self.lesson == 2:
            data = self.sample_lesson_2(index)
        elif self.lesson == 3:
            data = self.sample_lesson_3(index)
        else:
            data = None
        return data

    def load_kg(self, kg_ent_path, kg_prop_path, ent_path, rel_path, prop_path):
        # load kg data
        self.kg_ent_list = self.load_and_cache("kg-ent-list", self.read_triple_list, kg_ent_path)
        if os.path.exists(kg_prop_path):
            self.kg_prop_list = self.load_and_cache("kg-prop-list", self.read_triple_list, kg_prop_path)
        else:
            self.kg_prop_list = []
        self.ent_name = self.load_and_cache("ent-name", self.read_alias, ent_path)
        self.rel_name = self.load_and_cache("rel-name", self.read_alias, rel_path)
        if os.path.exists(prop_path):
            self.prop_name = self.load_and_cache("prop-name", self.read_alias, prop_path)
        else:
            self.prop_name = dict()
        
        # load multi-hop & multi-obj data for lesson 2 & 3
        self.mh_kg_list, self.mh_head_set, self.head_2_kg_ent, self.head_2_kg_prop, self.head_2_kg_mh = self.load_and_cache("multi-hop-kg", self.filter_multi_hop)
        self.mo_kg_list = self.load_and_cache("multi-obj-kg", self.filter_multi_obj)
        return

    def load_and_cache(self, name, func, *args, **kwargs):
        if not os.path.exists(self.cache_root):
            os.mkdir(self.cache_root)
        cached_path = os.path.join(self.cache_root, f"{name}.pkl")
        if os.path.exists(cached_path):
            logging.info(f"loading {cached_path} ...")
            with open(cached_path, "rb") as file:
                data = pickle.load(file)
        else:
            data = func(*args, **kwargs)
            logging.info(f"caching {cached_path} ...")
            with open(cached_path, "wb") as file:
                pickle.dump(data, file)
        return data
    
    def read_triple_list(self, path):
        # [(head, rel, tail)]
        kg_data = []
        with open(path, "rt", encoding="utf-8") as file:
            for line in file:
                head, rel, tail = line.strip('\n').split('\t')
                kg_data.append((head, rel, tail))
        return kg_data

    def read_alias(self, path):
        # ent -> [name]
        alias = dict()
        with open(path, "rt", encoding="utf-8") as file:
            for line in file:
                fields = line.strip('\n').split('\t')
                ent = fields[0]
                names = fields[1:]
                alias[ent] = names
        return alias
    
    def filter_multi_hop(self):
        # multi-hop: head1 -> tail1 (head2) -> tail2
        # head2 -> tail2 (kg_data)
        head_set = set()
        for head, rel, tail in self.kg_ent_list:
            head_set.add(head)
        for head, rel, tail in self.kg_prop_list:
            head_set.add(head)
        # head1 -> tail1 (mh_kg_list)
        # 2-hop: tail1 not in mh_head_set: head1 -> tail1 (head2) -> tail2
        # 3-hop: tail1 in mh_head_set: head1 -> tail1 (head1) -> tail1 (head2) -> tail2
        mh_kg_list = list()
        mh_head_set = set()
        for head, rel, tail in self.kg_ent_list:
            if tail in head_set:
                mh_kg_list.append((head, rel, tail))
                mh_head_set.add(head)
        
        # fetching tail given head in hop 2/3: head -> [(rel, tail)]
        # head2 -> tail2 (tail2 is entity)
        head_2_kg_ent = dict()
        for head, rel, tail in self.kg_ent_list:
            if head not in head_2_kg_ent.keys():
                head_2_kg_ent[head] = []
            head_2_kg_ent[head].append((rel, tail))
        # head2 -> tail2 (tail2 is property)
        head_2_kg_prop = dict()
        for head, rel, tail in self.kg_prop_list:
            if head not in head_2_kg_prop.keys():
                head_2_kg_prop[head] = []
            head_2_kg_prop[head].append((rel, tail))
        # head1 -> tail1 (for 3-hop)
        head_2_kg_mh = dict()
        for head, rel, tail in mh_kg_list:
            if head not in head_2_kg_mh.keys():
                head_2_kg_mh[head] = []
            head_2_kg_mh[head].append((rel, tail))
        return mh_kg_list, mh_head_set, head_2_kg_ent, head_2_kg_prop, head_2_kg_mh
    
    def filter_multi_obj(self):
        # multi-obj: head -> rel -> [tail]
        # multi-obj only exists in entity as tail
        kg_ent_dict = dict()
        for head, rel, tail in self.kg_ent_list:
            if head not in kg_ent_dict.keys():
                kg_ent_dict[head] = dict()
            if rel not in kg_ent_dict[head].keys():
                kg_ent_dict[head][rel] = set()
            if tail not in kg_ent_dict[head][rel]:
                kg_ent_dict[head][rel].add(tail)
        
        mo_kg_list = list()
        for head, rel_tails in kg_ent_dict.items():
            for rel, tails in rel_tails.items():
                if len(tails) > 1:
                    mo_kg_list.append((head, rel, list(tails)))
        return mo_kg_list
    
    def set_lesson(self, lesson):
        self.lesson = lesson
        return

    def split_words_cut(self, words, prop_idx_list, identical_ent_list):
        # split word segmentation for property, matching identical word index (identical word: all masked or all unmasked)
        # ["青蒿素", "性状", "无色 晶体", "无色 晶体"], [[2, 3]] => ["青蒿素", "性状", "无色", "晶体", "无色", "晶体"], [[2, 4], [3, 5]]
        if len(prop_idx_list) > 0:
            word_group_list = [[w for w in word.split(' ') if w != ''] if word_idx in prop_idx_list else [word] for word_idx, word in enumerate(words)]
            # new index of each splitted word
            # new index: values of list, old index: index of list
            word_idx_list = []
            word_num = 0
            for word_group in word_group_list:
                word_idx_list.append(list(range(word_num, word_num + len(word_group))))
                word_num += len(word_group)
            words = [word for word_group in word_group_list for word in word_group]
            # match identical word for new index, based on old index
            if identical_ent_list is not None:
                n_identical_ent_list = []
                for el in identical_ent_list:
                    sub_len = len(word_idx_list[el[0]])
                    for sub_idx in range(sub_len):
                        n_identical_ent_list.append([word_idx_list[edx][sub_idx] for edx in el])
                identical_ent_list = n_identical_ent_list
        return words, identical_ent_list

    def tokenize_align_words(self, words):
        # tokenize words, grouping index of tokens in the same word (for whole word masking)
        # ["青蒿素", "性状", "无色", "晶体"] => ["cls", "青", "蒿", "素", "性", "状", "无", "色", "晶", "体", "sep"], [[1, 2, 3], [4, 5], [6, 7], [8, 9]]
        aligned_word_token_list = []
        match_flag = True
        tokenized = self.tokenizer(self.sep.join(words), return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = tokenized.pop("offset_mapping").tolist()[0]
        
        if len(words) > 0:
            # whether each token belongs to current word, based on start & end of word & token
            # start & end index of each word in the join
            acc_word_start = 0
            word_start_end_list = []
            for word in words:
                word_start_end_list.append((acc_word_start, acc_word_start + len(word)))
                acc_word_start += len(word) + len(self.sep)
            temp_word_token = []
            word_idx = 0
            word_start, word_end = word_start_end_list[word_idx]
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                # skip <cls> & <sep>
                if token_end > 0:
                    # token in current word: token_start >= word_start & token_end <= word_end, token_start < word_end
                    # token in following word: token_start >= word_end
                    while word_idx < len(word_start_end_list) and token_start >= word_end:
                        # keep empty word to match word_idx in identical_ent_list
                        aligned_word_token_list.append(temp_word_token)
                        temp_word_token = []
                        word_idx += 1
                        if word_idx < len(word_start_end_list):
                            word_start, word_end = word_start_end_list[word_idx]
                    temp_word_token.append(token_idx)
                    # (never happen) token cross two word: token_start < word_end & token_end > word_end
                    if token_end > word_end:
                        match_flag = False
                    # (never happen) sep token between words: token_start < word_start
                    if token_start < word_start:
                        match_flag = False
            if len(temp_word_token) > 0:
                aligned_word_token_list.append(temp_word_token)
                word_idx += 1
            if word_idx < len(word_start_end_list):
                aligned_word_token_list.extend([[]] * (len(word_start_end_list) - word_idx))
            if len(aligned_word_token_list) != len(word_start_end_list):
                match_flag = False
                if len(aligned_word_token_list) > len(word_start_end_list):
                    aligned_word_token_list = aligned_word_token_list[:len(word_start_end_list)]
                else:
                    aligned_word_token_list.extend([[]] * (len(word_start_end_list) - len(aligned_word_token_list)))
        if not match_flag:
            input_tokens = self.tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
            logging.warning(f"{words} != {[' '.join(input_tokens[idx] for idx in word) for word in aligned_word_token_list]}")
        return tokenized, aligned_word_token_list

    def mask_input_label(self, input_ids, mlm_mask):
        # adapted from DataCollatorForWholeWordMask @ huggingface
        labels = input_ids.clone()
        probability_matrix = mlm_mask
        special_token_mask = torch.tensor([self.tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)], dtype=torch.bool)
        probability_matrix.masked_fill_(special_token_mask, value=0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.replace_probability)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = torch.bernoulli(torch.full(labels.shape, self.random_probability)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, labels

    def whole_word_mask(self, words, prop_idx_list, identical_ent_list=None):
        words, identical_ent_list = self.split_words_cut(words, prop_idx_list, identical_ent_list)
        tokenized, aligned_words = self.tokenize_align_words(words)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        wwm_list = []
        # group index of identical words (only one word if not identical)
        if identical_ent_list is not None:
            identical_ents = set([edx for el in identical_ent_list for edx in el])
            ent_group_list = [[edx] for edx in range(len(aligned_words)) if edx not in identical_ents] + identical_ent_list
        else:
            ent_group_list = [[edx] for edx in range(len(aligned_words))]
        # group index of tokens in identical words
        for ent_group in ent_group_list:
            input_idx_group = []
            for ent_idx in ent_group:
                input_idx_list = [idx for idx in aligned_words[ent_idx] if tokens[idx] not in self.tokenizer.all_special_tokens]
                if len(input_idx_list) > 0:
                    input_idx_group.extend(input_idx_list)
            if len(input_idx_group) > 0:
                wwm_list.append(input_idx_group)
        random.shuffle(wwm_list)
        mlm_length = int((len(tokenized.input_ids[0]) - 2) * self.mlm_probability)
        mlm_list = []
        for wwm_item in wwm_list:
            mlm_list.extend(wwm_item)
            if len(mlm_list) >= mlm_length:
                break
        mlm_list = set(mlm_list)
        mlm_mask = torch.tensor([[1 if idx in mlm_list else 0 for idx in range(len(tokenized.input_ids[0]))]])
        input_ids, labels = self.mask_input_label(tokenized.input_ids, mlm_mask)
        tokenized["input_ids"] = input_ids
        tokenized["labels"] = labels
        # for data_collator: tensor(1, len) -> list(len)
        for k, v in tokenized.items():
            tokenized[k] = v[0].tolist()
        if (labels != -100).sum().item() == 0:
            logging.warning(f"skip for non-mask: {words}")
            tokenized = None
        return tokenized

    def triple_idx_2_name(self, triple_idx, triple_type, identical_ent_list=None):
        # ent/rel/prop index -> sampled name
        # triple_type: -1: text, 0: rel, 1: ent, 2: prop
        assert len(triple_idx) == len(triple_type)
        if identical_ent_list is not None:
            # identical ent/rel/prop use the same name
            # ent_index -> min_index of identical ents
            identical_ent_dict = dict()
            for ent_list in identical_ent_list:
                min_pos = None
                # two identical groups contain the same ent
                for ent_pos in ent_list:
                    if ent_pos in identical_ent_dict.keys():
                        min_pos = identical_ent_dict[ent_pos]
                        break
                if min_pos is None:
                    min_pos = min(ent_list)
                for ent_pos in ent_list:
                    identical_ent_dict[ent_pos] = min_pos
        else:
            identical_ent_dict = None
        triple_name = []
        for token_pos, (token_idx, token_type) in enumerate(zip(triple_idx, triple_type)):
            if token_type == -1:
                token_name = token_idx
            else:
                if identical_ent_dict is not None and token_pos in identical_ent_dict.keys() and identical_ent_dict[token_pos] != token_pos:
                    token_name = triple_name[identical_ent_dict[token_pos]]
                else:
                    if token_type == 0:
                        token_names = self.rel_name[token_idx]
                    elif token_type == 1:
                        token_names = self.ent_name[token_idx]
                    elif token_type == 2:
                        token_names = self.prop_name[token_idx]
                    else:
                        token_names = None
                    if len(token_names) > 1:
                        token_name = token_names[random.randint(0, len(token_names) - 1)]
                    else:
                        token_name = token_names[0]
                    # control length of property and entity/relation name string
                    # property <= max_prop_len, entity/relation <= max_name_len
                    # zh: length of string, en: number of words (space-splitted string)
                    if self.lang == "zh":
                        tokens = token_name
                    else:
                        tokens = token_name.split(' ')
                    if (token_type == 0 or token_type == 1) and len(tokens) > self.max_name_len:
                        tokens = tokens[:self.max_name_len]
                    if token_type == 2 and len(tokens) > self.max_prop_len:
                        tokens = tokens[:self.max_prop_len]
                    if self.lang == "zh":
                        token_name = tokens
                    else:
                        token_name = ' '.join(tokens)
            triple_name.append(token_name)
        return triple_name

    def sample_lesson_1(self, index):
        ent_length = len(self.kg_ent_list)
        if index < ent_length:
            triple = self.kg_ent_list[index]
            tail_type = 1
            prop_idx_list = []
        else:
            triple = self.kg_prop_list[index - ent_length]
            tail_type = 2
            prop_idx_list = [2]
        triple_type = (1, 0, tail_type)
        # sample name
        words = self.triple_idx_2_name(triple, triple_type)
        # tokenize & mask
        try:
            sample = self.whole_word_mask(words, prop_idx_list)
        except Exception as e:
            logging.warning(f"{e} @ {words}")
            sample = None
        return sample
    
    def sample_next_hop(self, head, final_hop=True, skip_ent=None):
        # final_hop: head2 -> tail2 (head_2_kg_ent/prop)
        # otherwise: head1 -> tail1 (head_2_kg_mh)
        # skip_ent: skip previous ents to avoid circle hop
        if final_hop:
            ent_candidates = self.head_2_kg_ent.get(head, [])
            prop_candidates = self.head_2_kg_prop.get(head, [])
            if skip_ent is not None:
                sk_ent_candidates = [(rel, tail) for rel, tail in ent_candidates if tail not in skip_ent]
                if len(sk_ent_candidates) + len(prop_candidates) > 0:
                    ent_candidates = sk_ent_candidates
            candidates = ent_candidates + prop_candidates
            idx = random.randint(0, len(candidates) - 1)
            rel, tail = candidates[idx]
            if idx < len(ent_candidates):
                is_ent = True
            else:
                is_ent = False
        else:
            candidates = self.head_2_kg_mh[head]
            if skip_ent is not None:
                sk_candidates = [(rel, tail) for rel, tail in candidates if tail not in skip_ent]
                if len(sk_candidates) > 0:
                    candidates = sk_candidates
            rel, tail = candidates[random.randint(0, len(candidates) - 1)]
            is_ent = True
        hop = (head, rel, tail)
        return hop, is_ent

    def sample_multi_hop(self, index, ret_cot=True):
        # ret_cot == True: evidence + conclusion, otherwise: conclusion only
        # head1 -> tail1 (head2) -> tail2
        start_hop = self.mh_kg_list[index]  # (head, rel, tail)
        next_head = start_hop[2]
        skip_ent = [start_hop[0], next_head]
        if next_head in self.mh_head_set and random.random() > self.step_2_probability:
            step = 3
        else:
            step = 2
        if step == 3:
            mid_hop, _ = self.sample_next_hop(next_head, final_hop=False, skip_ent=skip_ent)
            next_head = mid_hop[2]
            skip_ent.append(next_head)
            mid_hops = [mid_hop]
        else:
            mid_hops = []
        end_hop, is_ent = self.sample_next_hop(next_head, final_hop=True, skip_ent=skip_ent)
        all_hops = [start_hop] + mid_hops + [end_hop]
        # type: -1: text, 0: rel, 1: ent, 2: prop
        ent_tail_type = 1 if is_ent else 2
        # conclusion: (h1 r1 r2 t2) or (h1 r1 r2 r3 t3)
        conclusion = [all_hops[0][0]] + [hop[1] for hop in all_hops] + [all_hops[-1][2]]
        conclusion_type = [1] + [0 for _ in all_hops] + [ent_tail_type]
        # evidence: (h1 r1 t1 <sep> h2 r2 t2 <sep>) or (h1 r1 t1 <sep> h2 r2 t2 <sep> h3 r3 t3 <sep>)
        if ret_cot:
            evidences = []
            evidences_type = []
            for hop in all_hops:
                evidences.extend(hop)
                evidences.append(self.period)
                evidences_type.extend([1, 0, 1, -1])
            evidences_type[-2] = ent_tail_type
            cot = evidences + conclusion
            cot_type = evidences_type + conclusion_type
        else:
            cot, cot_type = None, None
        # idx for prop in conclusion: idx_in_cot = idx_in_conclusion + evidence_len
        prop_idx_list = [] if is_ent else [len(conclusion) - 1]
        if ret_cot:
            con_prefix_len = len(evidences)
            prop_idx_list = [i + con_prefix_len for i in prop_idx_list]
            # idx for prop in evidence
            prop_idx_list += [] if is_ent else [len(evidences) - 2]
            # identical ent: h1_evi = h1_con, r1/r2/r3_evi = r1/r2/r3_con, t3_evi = t3_con
            # (h1 r1 t1 <sep> h2 r2 t2 <sep> h3 r3 t3 <sep>) <-> (h1 r1 r2 r3 t3)
            identical_list = [[0, 0]] + [[i * 4 + 1, i + 1] for i in range(len(all_hops))] + [[len(evidences) - 2, len(conclusion) - 1]]
            # idx_in_conclusion -> idx_in_cot
            for i in range(len(identical_list)):
                identical_list[i][1] += con_prefix_len
        else:
            identical_list = None
        return conclusion, conclusion_type, cot, cot_type, prop_idx_list, identical_list
    
    def sample_multi_obj(self, index, ret_cot=True):
        # ret_cot == True: evidence + conclusion, otherwise: conclusion only
        # head -> rel -> tail1 / tail2
        head, rel, tails = self.mo_kg_list[index]
        if len(tails) > 2 and random.random() > self.step_2_probability:
            step = 3
        else:
            step = 2
        sampled_tails = random.sample(tails, step)
        # conclusion: (h r t1 t2) or (h r t1 t2 t3)
        conclusion = [head, rel] + sampled_tails
        conclusion_type = [1, 0] + [1 for _ in sampled_tails]   # multi-obj only exists in entity as tail
        # evidence: (h r t1 <sep> h r t2 <sep>) or (h r t1 <sep> h r t2 <sep> h r t3 <sep>)
        if ret_cot:
            evidences = []
            evidences_type = []
            for tail in sampled_tails:
                evidences.extend((head, rel, tail, self.period))
                evidences_type.extend([1, 0, 1, -1])
            cot = evidences + conclusion
            cot_type = evidences_type + conclusion_type
        else:
            cot, cot_type = None, None
        prop_idx_list = []
        if ret_cot:
            # identical ent: h_evi = h_con, r_evi = r_con, t1/t2/t3_evi = t1/t2/t3_con
            # (h r t1 <sep> h r t2 <sep> h r t3 <sep>) <-> (h r t1 t2 t3)
            identical_list = [[i * 4, 0] for i in range(len(sampled_tails))] + [[i * 4 + 1, 1] for i in range(len(sampled_tails))] + [[i * 4 + 2, i + 2] for i in range(len(sampled_tails))]
            # idx_in_conclusion -> idx_in_cot
            con_prefix_len = len(evidences)
            for i in range(len(identical_list)):
                identical_list[i][1] += con_prefix_len
        else:
            identical_list = None
        return conclusion, conclusion_type, cot, cot_type, prop_idx_list, identical_list

    def sample_lesson_2(self, index):
        # cot: evidence + conclusion
        mh_length = len(self.mh_kg_list)
        if index < mh_length:
            _, _, cot, cot_type, prop_idx_list, identical_list = self.sample_multi_hop(index, ret_cot=True)
        else:
            _, _, cot, cot_type, prop_idx_list, identical_list = self.sample_multi_obj(index - mh_length, ret_cot=True)
        words = self.triple_idx_2_name(cot, cot_type, identical_list)
        try:
            sample = self.whole_word_mask(words, prop_idx_list, identical_list)
        except Exception as e:
            logging.warning(f"{e} @ {words}")
            sample = None
        return sample
    
    def sample_lesson_3(self, index):
        # conclusion only
        mh_length = len(self.mh_kg_list)
        if index < mh_length:
            conclusion, conclusion_type, _, _, prop_idx_list, _ = self.sample_multi_hop(index, ret_cot=False)
        else:
            conclusion, conclusion_type, _, _, prop_idx_list, _ = self.sample_multi_obj(index - mh_length, ret_cot=False)
        words = self.triple_idx_2_name(conclusion, conclusion_type)
        try:
            sample = self.whole_word_mask(words, prop_idx_list)
        except Exception as e:
            logging.warning(f"{e} @ {words}")
            sample = None
        return sample
    
    def collator(self, batch):
        batch = [item for item in batch if item is not None]
        collated_batch = self.bert_collator(batch)
        for k, v in collated_batch.items():
            if v.size(1) > self.bert_max_len:
                collated_batch[k] = torch.cat((v[:, :self.bert_max_len-1], v[:, -1:]), dim=-1)
        return collated_batch

class QADataset(data.Dataset):
    def __init__(self, tokenizer, path, single_label=False, max_prop_len=256, max_name_len=32, lang="zh"):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.single_label = single_label
        self.max_prop_len = max_prop_len
        self.max_name_len = max_name_len
        self.lang = lang
        self.dataset = load_qa_dataset(path)
        self.bert_collator = DataCollatorWithPadding(self.tokenizer, padding="longest", return_tensors="pt")
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        if "question_tokenized" not in data.keys():
            if self.single_label:
                answers = [data["answer"]]
            else:
                answers = data["answer"].split("|||")
            negative = data["negative"]
            candidates = answers + negative
            if self.single_label:
                data["labels"] = 0
            else:
                data["labels"] = [1] * len(answers) + [0] * len(negative)
            if data["is_ent"]:
                max_len = self.max_name_len
            else:
                max_len = self.max_prop_len
            if self.lang == "zh":
                candidates = [cand[:max_len] for cand in candidates]
            else:
                candidates = [' '.join(cand.split(' ')[:max_len]) for cand in candidates]
            question = data["question"]
            candidates = [f"{question} {self.tokenizer.sep_token} {cand}" for cand in candidates]
            data["candidates_tokenized"] = [self.tokenizer(cand) for cand in candidates]
        return data
    
    def collator(self, batch):
        ids = []
        ent_candidates_tokenized = []
        prop_candidates_tokenized = []
        labels = []
        type_map = []
        pos_map = []
        for data_item in batch:
            if "id" in data_item.keys():
                ids.append(data_item["id"])
            else:
                ids.append(None)
            candidates = data_item["candidates_tokenized"]
            # ques-cand matrix -> candidate list: process candidates with BERT in one go
            # candidate list -> ent_candidates & prop_candidates: reduce padding and faster computation, since property may be much longer than entity
            # pos_map: indicate ques-cand matrix, pos_map[i, j] = i-th question's j-th candidate index in ent/prop_candidates
            # type_map[i] = i-th question's candidates in which list
            if data_item["is_ent"]:
                group_type = 0
                group = ent_candidates_tokenized
            else:
                group_type = 1
                group = prop_candidates_tokenized
            type_map.append(group_type)
            pos_map.append([cand_idx + len(group) for cand_idx in range(len(candidates))])
            group.extend(candidates)
            labels.append(data_item["labels"])
        # collate tokenized data
        if len(ent_candidates_tokenized) > 0:
            ent_candidates_tokenized = self.bert_collator(ent_candidates_tokenized)
            if not hasattr(ent_candidates_tokenized, "token_type_ids"):
                ent_candidates_tokenized["token_type_ids"] = None
        else:
            ent_candidates_tokenized = None
        if len(prop_candidates_tokenized) > 0:
            prop_candidates_tokenized = self.bert_collator(prop_candidates_tokenized)
            if not hasattr(prop_candidates_tokenized, "token_type_ids"):
                prop_candidates_tokenized["token_type_ids"] = None
        else:
            prop_candidates_tokenized = None
        # pad pos_map & label matrix
        max_len = max(len(item) for item in pos_map)
        pos_map = [item + [-1] * (max_len - len(item)) for item in pos_map]
        if not self.single_label:
            labels = [item + [0] * (max_len - len(item)) for item in labels]
        batch_data = dict()
        batch_data["ids"] = ids
        batch_data["ent_candidates"] = ent_candidates_tokenized
        batch_data["prop_candidates"] = prop_candidates_tokenized
        batch_data["type_map"] = torch.tensor(type_map)
        batch_data["pos_map"] = torch.tensor(pos_map)
        batch_data["labels"] = torch.tensor(labels)
        return batch_data

class MWPDataset(data.Dataset):
    def __init__(self, tokenizer, path, class_list=None, max_expr_length=None):
        super(MWPDataset, self).__init__()
        self.tokenizer = tokenizer
        self.op_set = set("+-*/^")
        self.dataset = load_qa_dataset(path)
        if class_list is not None:
            self.class_list = class_list
            self.max_expr_length = max_expr_length
        else:
            self.class_list, self.max_expr_length = self.fetch_class_list()
        self.class_dict = {c:i for i, c in enumerate(self.class_list)}
        self.label_pad_id = self.class_dict[self.class_list[0]]
        self.bert_collator = DataCollatorWithPadding(self.tokenizer, padding="longest", return_tensors="pt")
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        if "tokenized" not in data.keys():
            tokenized = self.tokenizer(data["text"])
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized.input_ids)
            # input data: convert numbers in question into "n" instead of "temp_x" or original number to avoid mistakenly tokenized by Tokenizer
            # here: recognize each number ("n"), recover to "temp_x", and point out its position in the question with num_pos
            # value in pad/op/const_index: -1, num_index(temp_x): position of temp_x in the question
            num_pos = [-1] * len(self.class_list)
            num_idx = 0
            for pos, w in enumerate(tokens):
                if w == "n":
                    temp = "temp_" + chr(ord('a') + num_idx)
                    num_idx += 1
                    num_pos[self.class_dict[temp]] = pos
            labels = data["expr"].split(" ")
            labels = [self.class_dict[w] if w in self.class_dict.keys() else -1 for w in labels]
            data["tokenized"] = tokenized
            data["labels"] = labels
            data["num_pos"] = num_pos
        return data
    
    def collator(self, batch):
        questions = []
        labels = []
        num_poses = []
        answers = []
        num_lists = []
        for data_item in batch:
            questions.append(data_item["tokenized"])
            labels.append(data_item["labels"])
            num_poses.append(data_item["num_pos"])
            answers.append(data_item["answer"])
            # store original number for each temp_x to recover expression and compute final answer
            num_lists.append(data_item["num_list"])
        questions = self.bert_collator(questions)
        if not hasattr(questions, "token_type_ids"):
            questions["token_type_ids"] = None
        max_len = max(len(label) for label in labels)
        labels = [label + [self.label_pad_id] * (max_len - len(label)) for label in labels]
        batch_data = dict()
        batch_data["questions"] = questions
        batch_data["labels"] = torch.tensor(labels)
        batch_data["num_pos"] = torch.tensor(num_poses)
        batch_data["answers"] = answers
        batch_data["num_list"] = num_lists
        return batch_data

    def fetch_class_list(self):
        pad_token = "pad"
        num_len = max(len(data["num_list"]) for data in self.dataset)
        num_list = ["temp_" + chr(ord('a') + i) for i in range(num_len)]
        op_list = []
        const_list = []
        max_expr_length = 0
        for data in self.dataset:
            expr = data["expr"].split(' ')
            if len(expr) > max_expr_length:
                max_expr_length = len(expr)
            for w in expr:
                if not w.startswith("temp_"):
                    if w in self.op_set:
                        if w not in op_list:
                            op_list.append(w)
                    else:
                        if w not in const_list:
                            const_list.append(w)
        class_list = [pad_token] + op_list + const_list + num_list
        return class_list, max_expr_length
