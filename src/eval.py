import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from loguru import logger
from tqdm import trange

from .datasets.loader import get_dataloader
from .datasets.utils import Split
from .methods import __dict__ as all_methods
from .models.ingredient import get_model
from .train import parse_args
from .utils import (
    load_checkpoint,
    get_model_dir,
    copy_config,
)


def hash_config(args: argparse.Namespace) -> str:
    res = 0
    for i, (key, value) in enumerate(args.items()):
        if key != "port":
            if type(value) == str:
                hash_ = sum(value.encode())
            elif type(value) in [float, int, bool]:
                hash_ = round(value, 3)
            else:
                hash_ = sum(
                    [
                        int(v) if type(v) in [float, int, bool] else sum(v.encode())
                        for v in value
                    ]
                )
            res += hash_ * random.randint(1, int(1e6))

    return str(res)[-10:].split(".")[0]


def main_worker(args: argparse.Namespace) -> None:
    """
    Run the evaluation over all the tasks in parallel
    inputs:
        model : The loaded model containing the feature extractor
        loaders_dic : Dictionnary containing training and testing loaders
        model_path : Where was the model loaded from
        model_tag : Which model ('final' or 'best') to load
        method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
        shots : Number of support shots to try

    returns :
        results : List of the mean accuracy for each number of support shots
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"==> Setting up wandb with args {args}")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        config=args,
        job_type="eval",
        resume="allow",
        name=f"{args.test_source}-"
        f"{args.arch}-"
        f"{args.val_episodes}-"
        f"{args.num_ways}-"
        f"{args.method}-"
        f"{args.seed}",
    )
    # ===============> Setup directories for current exp. <=================
    # ======================================================================
    exp_root = Path(os.path.join(args.res_path, args.method))
    exp_root.mkdir(exist_ok=True, parents=True)
    exp_no = hash_config(args)
    exp_root = exp_root / str(exp_no)
    copy_config(args, exp_root)

    logger.info(f"==>  Saving all at {exp_root}")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # ===============> Load data <=================
    # =============================================
    current_split = "TEST" if args.eval_mode == "test" else "VALID"

    test_loader, num_classes_test = get_dataloader(
        args=args,
        source=args.test_source,
        batch_size=args.val_batch_size,
        split=Split[current_split],
        episodic=True,
        version=args.loader_version,
    )

    logger.info(
        f"{current_split} dataset: {args.test_source} ({num_classes_test} classes)"
    )

    # ===============> Load model <=================
    # ==============================================
    # num_classes = 5 if args.episodic_training else num_classes_base
    model = get_model(args=args, num_classes=5).to(device)

    logger.info(
        f"Number of model parameters: {sum(p.data.nelement() for p in model.parameters())}"
    )

    if (
        not args.load_from_timm
    ):  # then we load the model from a local checkpoint obtained through training
        model_path = get_model_dir(args=args)
        load_checkpoint(model=model, model_path=model_path)
    model.eval()

    # ===============> Define metrics <=================
    # ==================================================

    iter_loader = iter(test_loader)

    # ===============> Load method <=================
    # ===============================================
    method = all_methods[args.method](args=args).to(device)
    method.eval()

    # ===============> Run method <=================
    # ==============================================
    tqdm_bar = trange(int(args.val_episodes / args.val_batch_size))
    epoch_metrics = defaultdict(list)
    for i in tqdm_bar:
        # ======> Reload model checkpoint (some methods may modify model) <=======
        support, query, support_labels, query_labels = next(iter_loader)
        # logger.info(query_labels.size())
        support = support.to(device)
        query = query.to(device)
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)
        task_ids = (i * args.val_batch_size, (i + 1) * args.val_batch_size)
        episode_metrics = method(
            model=model,
            task_ids=task_ids,
            support=support,
            query=query,
            y_s=support_labels,
            y_q=query_labels,
            phase_name="test",
        )
        for name, value in episode_metrics.items():
            epoch_metrics[name].append(value)

        tqdm_bar.set_description(
            f"accuracy: {episode_metrics['test/accuracy_episode']}"
        )

    # ===============> Compute final metrics <=================
    # =========================================================
    logger.info("Computing final metrics...")
    final_metrics = {}
    for name, values in epoch_metrics.items():
        final_metrics[f"{name}-mean"] = torch.mean(torch.stack(values, dim=0))
        final_metrics[f"{name}-std"] = torch.std(torch.stack(values, dim=0))

    # ===============> Save results <=================
    # ===============================================
    logger.info("Saving results...")
    wandb.log(final_metrics)
    logger.info(f"Final metrics: {final_metrics}")
    wandb.finish()
    logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpus)

    main_worker(args=args)
