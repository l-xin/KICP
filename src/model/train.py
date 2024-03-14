# -*- coding: utf-8 -*-

import logging
import json
import torch
from torch.optim import AdamW
from torch.cuda import amp
from transformers import get_linear_schedule_with_warmup

from metrics import get_sum_f1, get_sum_em, get_sum_mwp_acc

def kg_train(args, model, dataloader):
    total_steps = len(dataloader) * args.epoch
    warm_steps = int(total_steps * args.warm_up)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.weight_decay > 0:
        parameter_groups = [
            {'params': [p for n, p in model.trained_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.trained_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        parameter_groups = [p for n, p in model.trained_parameters()]
    optimizer = AdamW(parameter_groups, lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)
    if args.fp16:
        scaler = amp.GradScaler()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    elif args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.restore:
        restore_train_state = torch.load(args.train_ckpt)
        model_state, optimizer_state, scheduler_state, scaler_state, start_epoch, global_step = restore_train_state
        if args.n_gpu > 1 or args.local_rank != -1:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if args.fp16:
            scaler.load_state_dict(scaler_state)
    else:
        start_epoch = 0
        global_step = 0

    for epoch in range(start_epoch, args.epoch):
        epoch += 1
        acc_size, acc_loss = 0, 0
        model.train()
        for step, batch in enumerate(dataloader):
            step += 1
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            optimizer.zero_grad()
            if args.fp16:
                with amp.autocast():
                    loss, _ = model(**batch)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = model(**batch)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                optimizer.step()
            scheduler.step()

            global_step += 1
            size = batch["input_ids"].size(0)
            acc_size += size
            acc_loss += loss.item() * size
            if global_step % args.log_step == 0:
                logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")
                acc_size, acc_loss = 0, 0
        if acc_size > 0:
            logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")
        
        # synchronize saving checkpoint
        if args.local_rank != -1 and args.local_rank != 0:
            torch.distributed.barrier()

        if args.local_rank == -1 or args.local_rank == 0:
            if args.n_gpu > 1 or args.local_rank != -1:
                model_saved = model.module
            else:
                model_saved = model
            model_saved.lm.save(args.lm_ckpt)
            model_saved.save(args.mlm_ckpt)
            scaler_state = scaler.state_dict() if args.fp16 else None
            train_state = (model_saved.state_dict(), optimizer.state_dict(), scheduler.state_dict(), scaler_state, epoch, global_step)
            torch.save(train_state, args.train_ckpt)
        
        if args.local_rank == 0:
            torch.distributed.barrier()
    return

def qa_train(args, model, train_dataloader, test_train_dataloader, test_dataloader):
    total_steps = len(train_dataloader) * args.epoch
    warm_steps = int(total_steps * args.warm_up)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.weight_decay > 0:
        parameter_groups = [
            {'params': [p for n, p in model.trained_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.trained_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        parameter_groups = [p for n, p in model.trained_parameters()]
    optimizer = AdamW(parameter_groups, lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)
    if args.fp16:
        scaler = amp.GradScaler()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    elif args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.restore:
        restore_train_state = torch.load(args.train_ckpt)
        model_state, optimizer_state, scheduler_state, scaler_state, start_epoch, global_step, metric, metric_info = restore_train_state
        if args.n_gpu > 1 or args.local_rank != -1:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if args.fp16:
            scaler.load_state_dict(scaler_state)
    else:
        start_epoch = 0
        global_step = 0
        metric, metric_info = 0, None

    for epoch in range(start_epoch, args.epoch):
        epoch += 1
        acc_size, acc_loss = 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            step += 1
            if batch["ent_candidates"] is not None:
                for k, v in batch["ent_candidates"].items():
                    if v is not None:
                        batch["ent_candidates"][k] = v.to(args.device)
            if batch["prop_candidates"] is not None:
                for k, v in batch["prop_candidates"].items():
                    if v is not None:
                        batch["prop_candidates"][k] = v.to(args.device)
            batch["type_map"] = batch["type_map"].to(args.device)
            batch["pos_map"] = batch["pos_map"].to(args.device)
            batch["labels"] = batch["labels"].to(args.device)
            batch.pop("ids")
            optimizer.zero_grad()
            if args.fp16:
                with amp.autocast():
                    _, loss = model(**batch)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(**batch)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                optimizer.step()
            scheduler.step()

            global_step += 1
            size = batch["type_map"].size(0)
            acc_size += size
            acc_loss += loss.item() * size
            if global_step % args.log_step == 0:
                logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")
                acc_size, acc_loss = 0, 0
        if acc_size > 0:
            logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")

        # synchronize evaluating & saving checkpoint
        if args.local_rank != -1 and args.local_rank != 0:
            torch.distributed.barrier()

        if args.local_rank == -1 or args.local_rank == 0:
            if args.n_gpu > 1 or args.local_rank != -1:
                model_saved = model.module
            else:
                model_saved = model

            f1, em = qa_evaluate(args, model_saved, test_train_dataloader)
            logging.info(f"train epoch {epoch}: f1 {f1:.3f}, em {em:.3f}")
            f1, em = qa_evaluate(args, model_saved, test_dataloader)
            logging.info(f"test epoch {epoch}: f1 {f1:.3f}, em {em:.3f}")
            if f1 > metric:
                metric = f1
                metric_info = (epoch, f1, em)
                torch.save(model_saved.state_dict(), args.model_ckpt)
                logging.info(f"saving model at epoch {epoch}")
            
            scaler_state = scaler.state_dict() if args.fp16 else None
            train_state = (model_saved.state_dict(), optimizer.state_dict(), scheduler.state_dict(), scaler_state, epoch, global_step, metric, metric_info)
            torch.save(train_state, args.train_ckpt)
        
        if args.local_rank == 0:
            torch.distributed.barrier()
    if (args.local_rank == -1 or args.local_rank == 0) and metric_info is not None:
        epoch, f1, em = metric_info
        logging.info(f"best at epoch {epoch}: f1 {f1:.3f}, em {em:.3f}")
    return

def qa_evaluate(args, model, dataloader):
    if args.best:
        model_state = torch.load(args.model_ckpt)
        model.load_state_dict(model_state)
    model.eval()
    sum_em = 0
    sum_f1 = 0
    sum_size = 0
    if args.test_log is not None:
        test_log = []
    for batch in dataloader:
        if batch["ent_candidates"] is not None:
            for k, v in batch["ent_candidates"].items():
                if v is not None:
                    batch["ent_candidates"][k] = v.to(args.device)
        if batch["prop_candidates"] is not None:
            for k, v in batch["prop_candidates"].items():
                if v is not None:
                    batch["prop_candidates"][k] = v.to(args.device)
        batch["type_map"] = batch["type_map"].to(args.device)
        batch["pos_map"] = batch["pos_map"].to(args.device)
        labels = batch.pop("labels").to(args.device)
        ids = batch.pop("ids")
        with torch.no_grad():
            if args.fp16:
                with amp.autocast():
                    scores, _ = model(**batch)
            else:
                scores, _ = model(**batch)
            batch_size = batch["type_map"].size(0)
            sum_size += batch_size
            if model.single_label:
                # pred = scores.topk(1)[1].squeeze(-1)
                # topk select 0 if all scores are equal in early training, leading to overestimation (0 is the true answer as processed in QADataset)
                pred = (torch.roll(scores, -1, dims=-1).topk(1)[1].squeeze(-1) + 1) % scores.size(-1)
                sum_f1 += (pred == labels).sum().item()
                if args.test_log is not None:
                    log_answer_list = pred.cpu().tolist()
            else:
                # pred = (scores > 0).int()
                # adjust threshold for unbalanced label distribution
                pred = (scores > 0.5).int()
                sum_f1 += get_sum_f1(pred, labels)
                sum_em += get_sum_em(pred, labels)
                if args.test_log is not None:
                    log_answer_list = [[idx for idx, v in enumerate(p) if v == 1] for p in pred.cpu().tolist()]
        if args.test_log is not None:
            for id, log_answer in zip(ids, log_answer_list):
                log_item = dict()
                log_item["id"] = id
                log_item["answer"] = log_answer
                test_log.append(log_item)
    f1 = sum_f1 / sum_size
    em = sum_em / sum_size
    if args.test_log is not None:
        with open(args.test_log, "wt", encoding="utf-8") as file:
            file.write('\n'.join(json.dumps(log_item, ensure_ascii=False) for log_item in test_log))
    return f1, em

def mwp_train(args, model, train_dataloader, test_train_dataloader, test_dataloader):
    total_steps = len(train_dataloader) * args.epoch
    warm_steps = int(total_steps * args.warm_up)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.weight_decay > 0:
        parameter_groups = [
            {'params': [p for n, p in model.trained_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.trained_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        parameter_groups = [p for n, p in model.trained_parameters()]
    optimizer = AdamW(parameter_groups, lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)
    if args.fp16:
        scaler = amp.GradScaler()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    elif args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.restore:
        restore_train_state = torch.load(args.train_ckpt)
        model_state, optimizer_state, scheduler_state, scaler_state, start_epoch, global_step, metric, metric_info = restore_train_state
        if args.n_gpu > 1 or args.local_rank != -1:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if args.fp16:
            scaler.load_state_dict(scaler_state)
    else:
        start_epoch = 0
        global_step = 0
        metric, metric_info = 0, None

    for epoch in range(start_epoch, args.epoch):
        epoch += 1
        acc_size, acc_loss = 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            step += 1
            for k, v in batch["questions"].items():
                if v is not None:
                    batch["questions"][k] = v.to(args.device)
            batch["num_pos"] = batch["num_pos"].to(args.device)
            batch["labels"] = batch["labels"].to(args.device)
            batch.pop("answers")
            batch.pop("num_list")
            optimizer.zero_grad()
            if args.fp16:
                with amp.autocast():
                    _, loss = model(**batch)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(**batch)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if args.max_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                optimizer.step()
            scheduler.step()

            global_step += 1
            size = batch["num_pos"].size(0)
            acc_size += size
            acc_loss += loss.item() * size
            if global_step % args.log_step == 0:
                logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")
                acc_size, acc_loss = 0, 0
        if acc_size > 0:
            logging.info(f"step {global_step} ({epoch}-{step}): {acc_loss/acc_size:.3f}")

        # synchronize evaluating & saving checkpoint
        if args.local_rank != -1 and args.local_rank != 0:
            torch.distributed.barrier()

        if args.local_rank == -1 or args.local_rank == 0:
            if args.n_gpu > 1 or args.local_rank != -1:
                model_saved = model.module
            else:
                model_saved = model

            acc = mwp_evaluate(args, model_saved, test_train_dataloader)
            logging.info(f"train epoch {epoch}: acc {acc:.3f}")
            acc = mwp_evaluate(args, model_saved, test_dataloader)
            logging.info(f"test epoch {epoch}: acc {acc:.3f}")
            if acc > metric:
                metric = acc
                metric_info = (epoch, acc)
                torch.save(model_saved.state_dict(), args.model_ckpt)
                logging.info(f"saving model at epoch {epoch}")
            
            scaler_state = scaler.state_dict() if args.fp16 else None
            train_state = (model_saved.state_dict(), optimizer.state_dict(), scheduler.state_dict(), scaler_state, epoch, global_step, metric, metric_info)
            torch.save(train_state, args.train_ckpt)
        
        if args.local_rank == 0:
            torch.distributed.barrier()
    if (args.local_rank == -1 or args.local_rank == 0) and metric_info is not None:
        epoch, acc = metric_info
        logging.info(f"best at epoch {epoch}: acc {acc:.3f}")
    return

def mwp_evaluate(args, model, dataloader):
    if args.best:
        model_state = torch.load(args.model_ckpt)
        model.load_state_dict(model_state)
    model.eval()
    sum_acc = 0
    sum_size = 0
    class_list = dataloader.dataset.class_list
    class_pad_idx = dataloader.dataset.label_pad_id
    for batch in dataloader:
        for k, v in batch["questions"].items():
            if v is not None:
                batch["questions"][k] = v.to(args.device)
        batch["num_pos"] = batch["num_pos"].to(args.device)
        batch.pop("labels")
        answers = batch.pop("answers")
        num_list = batch.pop("num_list")
        with torch.no_grad():
            if args.fp16:
                with amp.autocast():
                    seq, _ = model(**batch)
            else:
                seq, _ = model(**batch)
            batch_size = batch["num_pos"].size(0)
            sum_size += batch_size
            sum_acc += get_sum_mwp_acc(seq, answers, class_list, num_list, class_pad_idx)
    acc = sum_acc / sum_size
    return acc
