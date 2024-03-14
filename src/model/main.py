# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import os
import random
import torch

def get_args():
    parser = ArgumentParser(description="Knowledge-Injected Curriculum Pretraining Framework")

    parser.add_argument("--bert", type=str, default="bert-base-uncased")
    parser.add_argument("--adapter-size", type=int, default=None)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--lesson", type=int, default=1)
    parser.add_argument("--qa", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--best", action="store_true", default=False)
    parser.add_argument("--test-log", type=str, default=None)
    parser.add_argument("--tune-bert", action="store_true", default=False)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--max-prop", type=int, default=128)
    parser.add_argument("--max-name", type=int, default=5)
    parser.add_argument("--pos-weight", type=float, default=5)

    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--max-grad", type=float, default=1)

    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="0")
    parser.add_argument("--saved-lm", type=str, default=None)
    parser.add_argument("--saved-mlm", type=str, default=None)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--log-step", type=int, default=400)

    args = parser.parse_args()

    if args.dataset == "fbqa":
        args.dataset_root = "dataset/fbqa"
        args.single_label = True
        args.lang = "en"
    elif args.dataset == "cwq":
        args.dataset_root = "dataset/cwq"
        args.single_label = False
        args.lang = "en"
    elif args.dataset == "math23k":
        args.dataset_root = "dataset/math23k"
        args.lang = "zh"
    return args

def init_env():
    args = get_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    args.device = torch.device("cpu")
    args.n_gpu = 0
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        if torch.cuda.is_available():
            if args.local_rank == -1:
                args.device = torch.device("cuda")
                args.n_gpu = torch.cuda.device_count()
            else:
                torch.cuda.set_device(args.local_rank)
                args.device = torch.device("cuda", args.local_rank)
                torch.distributed.init_process_group(backend="nccl")
                args.n_gpu = 1
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    
    # limit torch threshold
    torch.set_num_threads(4)

    exp_root = os.path.join("experiments", args.dataset, args.tag)
    if not os.path.exists(exp_root):
        if args.local_rank == -1 or args.local_rank == 0:
            os.mkdir(exp_root)
    # synchronize os.mkdir
    if args.local_rank != -1:
        torch.distributed.barrier()
    args.lm_ckpt = os.path.join(exp_root, "lm.pt")
    args.mlm_ckpt = os.path.join(exp_root, "mlm.pt")
    args.model_ckpt = os.path.join(exp_root, "model.pt")
    args.train_ckpt = os.path.join(exp_root, "train.pt")
    if args.log is None:
        if args.local_rank != -1:
            log_filename = f"exp-r{args.local_rank}.log"
        else:
            log_filename = f"exp.log"
        args.log = os.path.join(exp_root, log_filename)
    
    if args.saved_lm is not None:
        args.saved_lm = os.path.join("experiments", args.dataset, args.saved_lm, "lm.pt")
    if args.saved_mlm is not None:
        args.saved_mlm = os.path.join("experiments", args.dataset, args.saved_mlm, "mlm.pt")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=args.log)
    logging.info(f"======================================== starting {args.dataset}-{args.tag} ========================================")
    logging.info('\n' + '\n'.join([f"\t{'['+k+']':20}\t{v}" for k, v in dict(args._get_kwargs()).items()]))
    return args

def cleanup(args):
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()
    return

# set cuda environment before import transformers
args = init_env()

from torch.utils import data
from transformers import BertTokenizer, BertTokenizerFast

from lm import KModel
from modules import MLM, QAModel, MWPModel
from dataset import KG2CorpusDataset, QADataset, MWPDataset
from train import kg_train, qa_train, mwp_train, qa_evaluate, mwp_evaluate

def pretrain(args):
    # synchronize downloading BERT model & tokenizer
    if args.local_rank != -1 and args.local_rank != 0 and args.cache:
        torch.distributed.barrier()
    
    tokenizer = BertTokenizerFast.from_pretrained(args.bert)
    if args.saved_mlm is not None:
        mlm = MLM.from_pretrained(args.saved_mlm)
    else:
        if args.saved_lm is not None:
            lm = KModel.from_pretrained(args.saved_lm, args.tune_bert)
        else:
            lm = KModel(args.bert, args.adapter_size, args.tune_bert)
        mlm = MLM(lm)
        mlm.init_cls(args.bert)

    if args.local_rank == 0 and args.cache:
        torch.distributed.barrier()

    mlm.to(args.device)

    # synchronize preprocessing & caching KG2CorpusDataset
    if args.local_rank != -1 and args.local_rank != 0 and args.cache:
        torch.distributed.barrier()
    
    kg_ent_path = os.path.join(args.dataset_root, "kg-ent.txt")
    kg_prop_path = os.path.join(args.dataset_root, "kg-prop.txt")
    ent_path = os.path.join(args.dataset_root, "ent.txt")
    rel_path = os.path.join(args.dataset_root, "rel.txt")
    prop_path = os.path.join(args.dataset_root, "prop.txt")
    dataset = KG2CorpusDataset(
        tokenizer = tokenizer,
        cache_root = args.dataset_root,
        lesson = args.lesson,
        max_prop_len = args.max_prop,
        max_name_len = args.max_name,
        mlm_probability = args.mlm_prob,
        kg_ent_path = kg_ent_path,
        kg_prop_path = kg_prop_path,
        ent_path = ent_path,
        rel_path = rel_path,
        prop_path = prop_path,
        lang = args.lang
    )

    if args.local_rank == 0 and args.cache:
        torch.distributed.barrier()

    sampler = data.DistributedSampler(dataset) if args.local_rank != -1 else data.RandomSampler(dataset)
    dp_size = 1 if args.n_gpu == 0 else args.n_gpu
    dataloader = data.DataLoader(dataset, sampler=sampler, batch_size=args.batch * dp_size, collate_fn=dataset.collator, pin_memory=True, num_workers=4)

    kg_train(args, mlm, dataloader)
    return

def train_qa(args):
    # synchronize downloading BERT model & tokenizer
    if args.local_rank != -1 and args.local_rank != 0 and args.cache:
        torch.distributed.barrier()
    
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    if args.saved_lm is not None:
        lm = KModel.from_pretrained(args.saved_lm, args.tune_bert)
    else:
        lm = KModel(args.bert, args.adapter_size, args.tune_bert)
    model = QAModel(lm, single_label=args.single_label, pos_weight=args.pos_weight)

    if args.local_rank == 0 and args.cache:
        torch.distributed.barrier()

    model.to(args.device)
    
    train_path = os.path.join(args.dataset_root, "train.json")
    test_path = os.path.join(args.dataset_root, "test.json")
    train_dataset = QADataset(tokenizer=tokenizer, path=train_path, single_label=args.single_label, max_prop_len=args.max_prop, max_name_len=args.max_name, lang=args.lang)
    sampler = data.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    dp_size = 1 if args.n_gpu == 0 else args.n_gpu
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch * dp_size, collate_fn=train_dataset.collator, pin_memory=True, num_workers=1)
    test_train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, collate_fn=train_dataset.collator, pin_memory=True, num_workers=1)
    test_dataset = QADataset(tokenizer=tokenizer, path=test_path, single_label=args.single_label, max_prop_len=args.max_prop, max_name_len=args.max_name, lang=args.lang)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collator, pin_memory=True, num_workers=1)

    if args.test:
        args.epoch = 0
        args.restore = True
    # train mode: train QAModel, test mode: load final QAModel
    qa_train(args, model, train_dataloader, test_train_dataloader, test_dataloader)
    if args.test:
        # if test with best parameters: load best QAModel here
        results = qa_evaluate(args, model, test_dataloader)
        logging.info(f"test results: {results}")
    return

def train_mwp(args):
    # synchronize downloading tokenizer
    if args.local_rank != -1 and args.local_rank != 0 and args.cache:
        torch.distributed.barrier()
    
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    
    if args.local_rank == 0 and args.cache:
        torch.distributed.barrier()

    train_path = os.path.join(args.dataset_root, "train.json")
    test_path = os.path.join(args.dataset_root, "test.json")
    train_dataset = MWPDataset(tokenizer=tokenizer, path=train_path)
    sampler = data.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    dp_size = 1 if args.n_gpu == 0 else args.n_gpu
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch * dp_size, collate_fn=train_dataset.collator, pin_memory=True, num_workers=1)
    test_train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, collate_fn=train_dataset.collator, pin_memory=True, num_workers=1)
    test_dataset = MWPDataset(tokenizer=tokenizer, path=test_path, class_list=train_dataset.class_list, max_expr_length=train_dataset.max_expr_length)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collator, pin_memory=True, num_workers=1)

    # synchronize downloading BERT model
    if args.local_rank != -1 and args.local_rank != 0 and args.cache:
        torch.distributed.barrier()
    
    if args.saved_lm is not None:
        lm = KModel.from_pretrained(args.saved_lm, args.tune_bert)
    else:
        lm = KModel(args.bert, args.adapter_size, args.tune_bert)
    model = MWPModel(lm, train_dataset.class_list, train_dataset.op_set, train_dataset.label_pad_id, train_dataset.max_expr_length)

    if args.local_rank == 0 and args.cache:
        torch.distributed.barrier()

    model.to(args.device)
    if args.test:
        args.epoch = 0
        args.restore = True
    # train mode: train QAModel, test mode: load final QAModel
    mwp_train(args, model, train_dataloader, test_train_dataloader, test_dataloader)
    if args.test:
        # if test with best parameters: load best QAModel here
        results = mwp_evaluate(args, model, test_dataloader)
        logging.info(f"test results: {results}")
    return

if __name__ == "__main__":
    # by default: pretrain with kg
    if not args.qa:
        pretrain(args)
    # qa: train on QA & MWP
    elif args.dataset == "math23k":
        train_mwp(args)
    else:
        train_qa(args)
    cleanup(args)
