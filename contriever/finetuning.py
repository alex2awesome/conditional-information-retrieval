# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import sys
import torch
import signal
# from torch.utils.tensorboard import SummaryWriter
import logging
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.options import Options
from src import data, dist_utils, utils, contriever, finetuning_data, inbatch

import train
import socket
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)

def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - local_rank
        - global_rank
        - world_size
    """
    is_slurm_job = 'SLURM_JOB_ID' in os.environ and not 'WORLD_SIZE' in os.environ
    has_local_rank = hasattr(params, 'local_rank')

    # SLURM job without torch.distributed.launch
    if is_slurm_job and has_local_rank:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])
        params.world_size = int(os.environ['SLURM_NTASKS'])

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.main_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.main_addr
        os.environ['MASTER_PORT'] = str(params.main_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)
        is_distributed = True


    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])

        is_distributed = True

    # local job (single GPU)
    else:
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        is_distributed = False

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if is_distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        #print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            #world_size=params.world_size,
            #rank=params.global_rank,
        )



def finetuning(opt, model, optimizer, scheduler, tokenizer, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    if hasattr(model, "module"):
        eval_model = model.module
    else:
        eval_model = model
    eval_model = eval_model.get_encoder()

    train_dataset = finetuning_data.Dataset(
        datapaths=opt.train_data,
        negative_ctxs=opt.negative_ctxs,
        negative_hard_ratio=opt.negative_hard_ratio,
        negative_hard_min_idx=opt.negative_hard_min_idx,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=True,
    )
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
    evaluate(opt, eval_model, tokenizer, tb_logger, step)

    epoch = 1

    model.train()
    prev_ids, prev_mask = None, None
    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        for i, batch in enumerate(train_dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            step += 1

            train_loss, iter_stats = model(**batch, stats_prefix="train")
            train_loss.backward()

            if opt.optim == "sam" or opt.optim == "asam":
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:

                train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
                evaluate(opt, eval_model, tokenizer, tb_logger, step)

                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0:
                    utils.save(
                        eval_model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"step-{step}",
                    )
                model.train()

            if step >= opt.total_steps:
                break

        epoch += 1


def evaluate(opt, model, tokenizer, tb_logger, step):
    dataset = finetuning_data.Dataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=False,
    )
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    model.eval()
    if hasattr(model, "module"):
        model = model.module
    correct_samples, total_samples, total_step = 0, 0, 0
    all_q, all_g, all_n = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

            q_emb = model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=opt.norm_query)
            all_emb = model(input_ids=all_tokens, attention_mask=all_mask, normalize=opt.norm_doc)

            g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

            all_q.append(q_emb)
            all_g.append(g_emb)
            all_n.append(n_emb)

        all_q = torch.cat(all_q, dim=0)
        all_g = torch.cat(all_g, dim=0)
        all_n = torch.cat(all_n, dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        message = []
        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            logger.info(" | ".join(message))
            if tb_logger is not None:
                tb_logger.add_scalar(f"eval_acc", acc, step)
                tb_logger.add_scalar(f"mrr", mrr, step)


def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    opt.retriever_model_id = retriever_model_id
    model = inbatch.InBatch(opt, retriever, tokenizer)

    model = model.cuda()

    optimizer, scheduler = utils.set_optim(opt, model)
    # if dist_utils.is_main():
    #    utils.save(model, optimizer, scheduler, global_step, 0., opt, opt.output_dir, f"step-{0}")
    logger.info(utils.get_parameters(model))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)


if __name__ == "__main__":
    main()
