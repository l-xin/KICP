# -*- coding: utf-8 -*-

import json
import random

# sample entities from KG as negative candidates for KGQA
# prefer hard negatives with the same type (relation) as true answer
# e.g., (x, height, <true>) => (y, height, <negative>)

def read_kg_list(path):
    # [(subj, rel, obj)]
    kg = list()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            subj, rel, obj = line.strip('\n').split('\t')
            kg.append((subj, rel, obj))
    return kg

def read_ent_names(path):
    # ent_id -> [ent_name]
    names = dict()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            fields = line.strip('\n').split('\t')
            names[fields[0]] = fields[1:]
    return names

def get_obj_rel_dict(kg):
    # obj-rel mapping for finding <negative> with the same rel as <true>
    # <true> obj -> rel -> obj <negative>

    # obj -> [rel]
    obj_rel_dict = dict()
    # rel -> [obj]
    rel_obj_dict = dict()
    for subj, rel, obj in kg:
        if rel not in rel_obj_dict.keys():
            rel_obj_dict[rel] = set()
        rel_obj_dict[rel].add(obj)
        if obj not in obj_rel_dict.keys():
            obj_rel_dict[obj] = set()
        obj_rel_dict[obj].add(rel)
    for k, v in obj_rel_dict.items():
        obj_rel_dict[k] = list(v)
    for k, v in rel_obj_dict.items():
        rel_obj_dict[k] = list(v)
    return rel_obj_dict, obj_rel_dict

def negative_sample(in_paths, out_paths, kg_path, ent_path, n_sample=10, min_sample=2, sample_names=True):
    # n_sample: preferred number of candidates (including true answers)
    # min_sample: minimum number of sampled negatives (if too many true answers)
    temp_sample_size = n_sample * 2
    kg = read_kg_list(kg_path)
    rel_obj_dict, obj_rel_dict = get_obj_rel_dict(kg)
    ent_names = read_ent_names(ent_path)
    all_ent_list = list(ent_names.keys())
    for in_path, out_path in zip(in_paths, out_paths):
        all_cnt, err_cnt = 0, 0
        out_data = []
        with open(in_path, "rt", encoding="utf-8") as file:
            for line in file:
                all_cnt += 1
                js = json.loads(line)
                answer_list = set(js["answer"].split("|||"))
                ans_len = len(answer_list)
                answer_ids = set(js["answer_ids"])
                # preferred: hard negatives with the same relation as true answer
                # find relations of true answers
                rel_list = set()
                cand_ent_idx_list = set()
                for ans in answer_ids:
                    if ans in obj_rel_dict.keys():
                        rel_list.update(obj_rel_dict[ans])
                # find negative obj_idx with the same relations
                for rel in rel_list:
                    ent_list = rel_obj_dict[rel]
                    # limit number of initial samples to faster process
                    if len(ent_list) > temp_sample_size:
                        ent_list = random.sample(ent_list, temp_sample_size)
                    # filter possibly correct samples with the same name as the true answer
                    valid_ent_list = []
                    for ent in ent_list:
                        find_flag = False
                        for name in ent_names[ent]:
                            if name in answer_list:
                                find_flag = True
                                break
                        if not find_flag:
                            valid_ent_list.append(ent)
                    cand_ent_idx_list.update(valid_ent_list)
                cand_ent_idx_list = list(cand_ent_idx_list - answer_ids)    # just in case
                if len(cand_ent_idx_list) > temp_sample_size:
                    cand_ent_idx_list = random.sample(cand_ent_idx_list, temp_sample_size)
                # sample name for the negative obj_idx (random one or first one)
                if sample_names:
                    cand_ent_name_list = list(set([random.sample(ent_names[idx], 1)[0] for idx in cand_ent_idx_list]))
                else:
                    cand_ent_name_list = list(set(ent_names[idx][0] for idx in cand_ent_idx_list))
                cand_ent_name_list = [item for item in cand_ent_name_list if item not in answer_list]   # just in case
                # sample n negatives as final candidates
                data_sample = n_sample - ans_len
                if data_sample < min_sample:
                    data_sample = min_sample
                if data_sample > len(cand_ent_name_list):
                    data_sample = len(cand_ent_name_list)
                sampled_list = random.sample(cand_ent_name_list, data_sample)
                # insufficient hard negatives: randomly sample the rest negatives
                if len(sampled_list) + ans_len < n_sample:
                    err_cnt += 1
                    miss_cnt = n_sample - len(sampled_list) - ans_len
                    # randomly sample ent_idx from all ents
                    add_cand_ent_idx_list = random.sample(all_ent_list, temp_sample_size)
                    # filter true answer, existing negatives, and possibly correct / existing one with the same name
                    valid_cand_ent_idx_list = []
                    for cand_ent in add_cand_ent_idx_list:
                        if cand_ent not in answer_ids and cand_ent not in cand_ent_idx_list:
                            find_flag = False
                            for name in ent_names[cand_ent]:
                                if name in answer_list or name in sampled_list:
                                    find_flag = True
                                    break
                            if not find_flag:
                                valid_cand_ent_idx_list.append(cand_ent)
                    # sample name for the negative ent_idx
                    if sample_names:
                        add_cand_ent_name_list = list(set([random.sample(ent_names[idx], 1)[0] for idx in valid_cand_ent_idx_list]))
                    else:
                        add_cand_ent_name_list = list(set(ent_names[idx][0] for idx in valid_cand_ent_idx_list))
                    # sample the rest negatives as final candidates
                    if len(add_cand_ent_name_list) < miss_cnt:
                        miss_cnt = len(add_cand_ent_name_list)
                    add_sampled_list = random.sample(add_cand_ent_name_list, miss_cnt)
                    for item in add_sampled_list:
                        if item not in sampled_list:
                            sampled_list.append(item)
                js["negative"] = sampled_list
                out_data.append(json.dumps(js, ensure_ascii=False))
        with open(out_path, "wt", encoding="utf-8") as file:
            file.write('\n'.join(out_data))
        print("err_cnt", err_cnt, all_cnt, err_cnt/all_cnt)
    return

if __name__ == "__main__":
    # fbqa & cwq share the same kg: wikidata
    kg_path = "dataset/fbqa/kg-ent.txt"
    ent_path = "dataset/fbqa/ent.txt"

    # process cwq
    paths = [f"dataset/cwq/{d}.json" for d in ("train", "test")]
    negative_sample(paths, paths, kg_path, ent_path, n_sample=10, min_sample=2, sample_names=False)

    # process fbqa
    paths = [f"dataset/fbqa/{d}.json" for d in ("train", "test", "dev")]
    negative_sample(paths, paths, kg_path, ent_path, n_sample=10, min_sample=2, sample_names=True)
