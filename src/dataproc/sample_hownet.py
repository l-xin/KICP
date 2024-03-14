# -*- coding: utf-8 -*-

import json
from OpenHowNet import HowNetDict

# sample subgraph from HowNet based on the vocab of Math23K as the KG
# based on the Tsinghua OpenHowNet
# https://github.com/thunlp/OpenHowNet

# entity: <sense>, <sememe> (similar to "thing" & "category")
# relation: sense, sememe, other relations between sememe
# word => <sense>
# <sense> --(sememe)-> <sememe>
# <sememe> --(sense)-> <sense>
# <sememe> --(others)-> <sememe>

def fetch_vocab(paths, vocab_path):
    vocab_set = set()
    for path in paths:
        with open(path, "rt", encoding="utf-8") as file:
            js = json.load(file)
        for item in js:
            ques = item["text"].split(' ')
            for w in ques:
                # "n" is a special placeholder for real number
                if w != "n":
                    vocab_set.add(w)
    with open(vocab_path, "wt", encoding="utf-8") as file:
        json.dump(list(vocab_set), file, ensure_ascii=False, indent=4)
    return

def sample_hownet(vocab_path, ent_path, rel_path, kg_path):
    with open(vocab_path, "rt", encoding="utf-8") as file:
        vocab = json.load(file)
    hownet = HowNetDict()
    # No.sense: defined in HowNet, No.sememe: > all No.sense to avoid conflict
    sememe_idx_prefix = max(int(s.No) for s in hownet.get_all_senses()) + 1
    print(sememe_idx_prefix, len(hownet.get_all_senses()))
    # ent idx & name mapping
    sense_name_dict = dict()
    sememe_name_dict = dict()
    sememe_idx_dict = dict()
    triples = set()
    all_cnt, hit_cnt = 0, 0
    # whether one ent is processed to avoid repeated processing
    sense_expand = set()
    sememe_expand_sense = set()
    sememe_expand_sememe = set()
    # saving all sampled <sense> to complete sense-sememe relation in complete_sense_2_sememe_edge function
    all_senses = []
    for word in vocab:
        all_cnt += 1
        # word => <sense>
        sense_list = hownet.get_sense(word, language="zh")
        if len(sense_list) > 0:
            hit_cnt += 1
        for sense in sense_list:
            s_id = int(sense.No)
            assert s_id < sememe_idx_prefix, sense
            if s_id not in sense_expand:
                sense_expand.add(s_id)
                s_name = sense.zh_word
                if s_id not in sense_name_dict.keys():
                    sense_name_dict[s_id] = s_name
                    all_senses.append(sense)
                # <sense> -> <sememe>
                for sememe in sense.get_sememe_list():
                    se_name = sememe.zh
                    if se_name not in sememe_idx_dict.keys():
                        se_id = len(sememe_name_dict) + sememe_idx_prefix
                        sememe_name_dict[se_id] = se_name
                        sememe_idx_dict[se_name] = se_id
                    else:
                        se_id = sememe_idx_dict[se_name]
                    triples.add((s_id, "sememe", se_id))
                    # <sense> -> <sememe> -> <sense>
                    if se_id not in sememe_expand_sense:
                        sememe_expand_sense.add(se_id)
                        for r_sense in sememe.get_senses():
                            rs_id = int(r_sense.No)
                            assert rs_id < sememe_idx_prefix, r_sense
                            rs_name = r_sense.zh_word
                            if rs_id not in sense_name_dict.keys():
                                sense_name_dict[rs_id] = rs_name
                                all_senses.append(r_sense)
                            triples.add((se_id, "sense", rs_id))
                    # <sense> -> <sememe> -> <sememe>
                    if se_id not in sememe_expand_sememe:
                        sememe_expand_sememe.add(se_id)
                        for _, rel, r_sememe in sememe.get_related_sememes(return_triples=True):
                            rse_name = r_sememe.zh
                            if rse_name not in sememe_idx_dict.keys():
                                rse_id = len(sememe_name_dict) + sememe_idx_prefix
                                sememe_name_dict[rse_id] = rse_name
                                sememe_idx_dict[rse_name] = rse_id
                            else:
                                rse_id = sememe_idx_dict[rse_name]
                            triples.add((se_id, rel, rse_id))
                            # <sense> -> <sememe> -> <sememe> -> <sense>
                            if rse_id not in sememe_expand_sense:
                                sememe_expand_sense.add(rse_id)
                                for r_sense in r_sememe.get_senses():
                                    rs_id = int(r_sense.No)
                                    assert rs_id < sememe_idx_prefix, r_sense
                                    rs_name = r_sense.zh_word
                                    if rs_id not in sense_name_dict.keys():
                                        sense_name_dict[rs_id] = rs_name
                                        all_senses.append(r_sense)
                                    triples.add((rse_id, "sense", rs_id))
    # merge <sense> & <sememe> ent
    all_name_dict = dict()
    for k, v in sense_name_dict.items():
        all_name_dict[k] = v
    for k, v in sememe_name_dict.items():
        all_name_dict[k] = v
    with open(ent_path, "wt", encoding="utf-8") as file:
        for idx, name in all_name_dict.items():
            file.write(f"{idx}\t{name}\n")
    # map rel name to rel_idx & save kg
    rel_dict = dict()
    with open(kg_path, "wt", encoding="utf-8") as file:
        for subj, rel, obj in triples:
            if rel in rel_dict.keys():
                rel_idx = rel_dict[rel]
            else:
                rel_idx = len(rel_dict)
                rel_dict[rel] = rel_idx
            file.write(f"{subj}\t{rel_idx}\t{obj}\n")
    with open(rel_path, "wt", encoding="utf-8") as file:
        for name, idx in rel_dict.items():
            file.write(f"{idx}\t{name}\n")
    print(len(sense_name_dict))
    print(len(sememe_name_dict))
    print(len(sememe_idx_dict))
    print(len(all_name_dict))
    assert len(sense_name_dict) + len(sememe_name_dict) == len(all_name_dict)
    print(len(rel_dict))
    print(len(triples))
    print(all_cnt, hit_cnt)
    return all_senses

def read_idx_name(path):
    name_dict = dict()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            idx, name = line.strip('\n').split('\t')
            name_dict[idx] = name
    return name_dict

def read_kg_set(path):
    kg = set()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            subj, rel, obj = line.strip('\n').split('\t')
            kg.add((subj, rel, obj))
    return kg

def complete_sense_2_sememe_edge(all_senses, ent_path, rel_path, kg_path, out_kg_path):
    # sample_hownet: already saved complete sememe-sense relation for each sampled <sememe>
    # complete_sense_2_sememe_edge: to complete missing sense-sememe relation for each sampled <sense> for self-contained subgraph
    kg = read_kg_set(kg_path)
    ent_name = read_idx_name(ent_path)
    rel_idx = {v:k for k, v in read_idx_name(rel_path).items()}
    sememe_rel = rel_idx["sememe"]
    hownet = HowNetDict()
    sememe_idx_prefix = max(int(s.No) for s in hownet.get_all_senses()) + 1
    sememe_idx_dict = {name:idx for idx, name in ent_name.items() if int(idx) >= sememe_idx_prefix}
    add_kg = set()
    for sense in all_senses:
        s_id = int(sense.No)
        # <sense> -> <sememe>
        sememes = [sememe.zh for sememe in sense.get_sememe_list()]
        for sememe in sememes:
            # only take sampled <sememe>
            if sememe in sememe_idx_dict.keys():
                se_id = sememe_idx_dict[sememe]
                triple = (s_id, sememe_rel, se_id)
                if triple not in kg and triple not in add_kg:
                    add_kg.add(triple)
    print(len(add_kg))
    print(len(kg))
    all_kg = kg | add_kg
    print(len(all_kg))
    assert len(add_kg) + len(kg) == len(all_kg)
    with open(out_kg_path, "wt", encoding="utf-8") as file:
        for subj, rel, obj in all_kg:
            file.write(f"{subj}\t{rel}\t{obj}\n")
    return

def build_kg(vocab_path, ent_path, rel_path, kg_path, all_kg_path):
    all_senses = sample_hownet(vocab_path, ent_path, rel_path, kg_path)
    complete_sense_2_sememe_edge(all_senses, ent_path, rel_path, kg_path, all_kg_path)
    return

if __name__ == "__main__":
    train_path = "dataset/math23k/train.json"
    test_path = "dataset/math23k/test.json"
    dev_path = "dataset/math23k/dev.json"
    dataset_paths = [train_path, test_path, dev_path]
    vocab_path = "dataset/math23k/vocab.json"
    ent_path = "dataset/math23k/ent.txt"
    rel_path = "dataset/math23k/rel.txt"
    kg_path = "dataset/math23k/kg-ent-sub.txt"
    all_kg_path = "dataset/math23k/kg-ent.txt"

    fetch_vocab(dataset_paths, vocab_path)
    build_kg(vocab_path, ent_path, rel_path, kg_path, all_kg_path)
