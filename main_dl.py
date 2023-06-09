# general python imports

import datetime
import json
import math
import os
import time
from argparse import ArgumentParser
from typing import *

# general DL imports
import albumentations as A
import numpy as np

# torch imports
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from config import get_config

# SSL imports
from configs.cvt_config import CvT_CONFIG
from configs.data_config import DATASET
from data import build_loader
from deeplenseutils.augmentation import *
from deeplenseutils.dataset import DefaultDatasetSetup
from deeplenseutils.inference import Inference
from deeplenseutils.train import train
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import (
    auto_resume_helper,
    get_grad_norm,
    load_checkpoint,
    reduce_tensor,
    save_checkpoint,
)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    parser.add_argument(
        "--dataset_name",
        metavar="Model_X",
        type=str,
        default="Model_I",
        choices=["Model_I", "Model_II", "Model_III", "Model_IV"],
        help="dataset type for DeepLense project",
    )
    parser.add_argument(
        "--save",
        metavar="XXX/YYY",
        type=str,
        default="data",
        help="destination of dataset",
    )

    parser.add_argument("--num_workers", metavar="1", type=int, default=1, help="number of workers")

    parser.add_argument(
        "--train_config",
        type=str,
        default="CvT",
        help="transformer config",
        choices=[
            "CvT",
            "CCT",
            "TwinsSVT",
            "LeViT",
            "CaiT",
            "CrossViT",
            "PiT",
            "Swin",
            "T2TViT",
            "CrossFormer",
        ],
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    # parser.add_argument("--cuda", action="store_true", help="whether to use cuda")
    # parser.add_argument(
    #     "--no-cuda", dest="cuda", action="store_false", help="not to use cuda"
    # )
    # parser.set_defaults(cuda=True)

    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="local rank for DistributedDataParallel",
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples_1, samples_2, targets) in enumerate(data_loader):
        samples_1 = samples_1.cuda(non_blocking=True)
        samples_2 = samples_2.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        loss = model(samples_1, samples_2)

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(config, train_loader, trainset):
    dataset_train = trainset
    data_loader_train = train_loader

    config.defrost()
    config.DATA.TRAINING_IMAGES = len(dataset_train)
    config.freeze()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    # )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        _ = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start self-supervised pre-training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        # if dist.get_rank() == 0 and (
        #     epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        # ):
        #     save_checkpoint(
        #         config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger
        #     )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


if __name__ == "__main__":
    _, config = parse_option()
    print("config:", config)

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["GROUP"] = "1"
    distributed_training = False

    setup(rank=0, world_size=1)

    dataset_setup = DefaultDatasetSetup()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    if distributed_training:
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        torch.distributed.barrier()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    # deeplense logger
    os.makedirs("logger", exist_ok=True)
    logger = create_logger(output_dir="logger", name="swin_tiny_patch4_window7_224")

    trainset = dataset_setup.get_default_trainset()
    valset = dataset_setup.get_default_testset()

    batch_size = 64
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    main(config, train_loader=train_loader, trainset=trainset)
