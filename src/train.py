import argparse
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from torch import tensor, Tensor
from torch.backends import cudnn
from tqdm import tqdm

from .datasets.loader import get_dataloader
from .datasets.utils import Split
from .losses import __losses__
from .methods import FewShotMethod
from .methods import __dict__ as all_methods
from .models.ingredient import get_model
from .optim import get_optimizer, get_scheduler
from .utils import (
    AverageMeter,
    save_checkpoint,
    get_model_dir,
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
    copy_config,
    make_episode_visualization,
    parse_args,
)


def meta_val(
    args: argparse.Namespace,
    model: torch.nn.Module,
    method: FewShotMethod,
    val_loader: torch.utils.data.DataLoader,
) -> Tuple[Tensor, Tensor]:
    # Device
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    model.eval()
    method.eval()

    # Metrics
    episode_acc = tensor([0.0], device=device)

    total_episodes = int(args.val_episodes / args.val_batch_size)
    tqdm_bar = tqdm(val_loader, total=total_episodes)
    for i, (support, query, support_labels, query_labels) in enumerate(tqdm_bar):
        if i >= total_episodes:
            break

        y_s = support_labels.to(device)
        y_q = query_labels.to(device)

        _, soft_preds_q = method(
            model=model, support=support, query=query, y_s=y_s, y_q=y_q
        )

        soft_preds_q = soft_preds_q.to(device).detach()
        episode_acc += (soft_preds_q.argmax(-1) == y_q).float().mean()

        tqdm_bar.set_description(
            "Acc {:.2f}".format((episode_acc / (i + 1) * 100).item())
        )

    n_episodes = tensor(total_episodes, device=device)

    model.train()
    method.train()

    return episode_acc, n_episodes


def main_worker(args: argparse.Namespace) -> None:
    logger.info(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # ============ Define loaders ================

    train_loader, num_classes = get_dataloader(
        args=args,
        source=args.base_source,
        batch_size=args.batch_size,
        split=Split["TRAIN"],
        episodic=args.episodic_training,
        version=args.loader_version,
    )

    val_loader, _ = get_dataloader(
        args=args,
        source=args.val_source,
        batch_size=args.val_batch_size,
        split=Split["VALID"],
        episodic=True,
        version=args.loader_version,
    )

    # ============ Define model ================

    num_classes = args.num_ways if args.episodic_training else num_classes
    model = get_model(args=args, num_classes=num_classes).to(device)

    logger.info(
        f"Number of model parameters: "
        f"{sum(p.data.nelement() for p in model.parameters())}"
    )

    exp_dir = get_model_dir(args)
    copy_config(args, exp_dir)

    # ============ Define metrics ================
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()

    metrics: Dict[str, Tensor] = {
        "train_loss": torch.zeros(
            args.num_updates // args.print_freq, dtype=torch.float32
        ),
        "val_acc": torch.zeros(args.num_updates // args.eval_freq, dtype=torch.float32),
    }

    # ============ Optimizer ================

    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(args=args, optimizer=optimizer)
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape} {param.requires_grad}")

    # ============ Method ================

    method = all_methods[args.method](args=args)
    if not args.episodic_training:
        if args.loss not in __losses__:
            raise ValueError(f"Please set the loss among : {list(__losses__.keys())}")
        loss_fn = __losses__[args.loss]
        loss_fn = loss_fn(args=args, num_classes=num_classes, reduction="none")
    eval_fn = partial(
        meta_val, method=method, val_loader=val_loader, model=model, args=args
    )

    # ============ Start training ================
    model.train()
    method.train()

    best_val_acc1 = 0.0  # noqa: F841
    tqdm_bar = tqdm(train_loader, total=args.num_updates)
    for i, data in enumerate(tqdm_bar):
        if i >= args.num_updates:
            break

        # ======== Forward / Backward pass =========

        if args.episodic_training:
            support, query, support_labels, target = data
            support, support_labels = support.to(device), support_labels.to(device)
            query, target = query.to(device), target.to(device)

            loss, soft_preds = method(
                support=support,
                query=query,
                y_s=support_labels,
                y_q=target,
                model=model,
            )  # [batch, q_shot]
        else:
            (input_, target) = data
            input_, target = input_.to(device), target.to(device).long()
            loss, soft_preds = loss_fn(input_, target, model)

        train_acc.update(
            (soft_preds.argmax(-1) == target).float().mean().item(), i == 0
        )
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # ============ Log metrics ============
        b_size = tensor([args.batch_size]).to(device)  # type: ignore

        loss = loss.sum() / b_size
        losses.update(loss.item(), i == 0)

        # ============ Validation ============
        if i % args.eval_freq == 0:
            val_acc, n_episodes = eval_fn()

            val_acc /= n_episodes
            is_best = val_acc > best_val_acc1
            best_val_acc1 = max(val_acc, best_val_acc1)

            save_checkpoint(
                state={
                    "iter": i,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_val_acc1,
                },
                is_best=is_best,
                folder=exp_dir,
            )

            for k in metrics:
                if "val" in k:
                    metrics[k][int(i / args.eval_freq)] = eval(k)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpus)
    if args.debug:
        args.batch_size = 16 if not args.episodic_training else 1
        args.val_episodes = 10
    world_size = len(args.gpus)
    distributed = world_size > 1
    args.world_size = world_size
    args.distributed = distributed

    main_worker(args)
