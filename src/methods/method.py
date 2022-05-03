import argparse
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..metrics import Metric
import wandb

from ..metrics.lamba_metrics import accuracy, cross_entropy_loss


def collect_episode_metrics(query_logits, query_targets, phase_name):
    episode_metrics = defaultdict(float)

    episode_metrics[f"{phase_name}/accuracy_episode"] = accuracy(
        query_logits, query_targets
    )

    episode_metrics[f"{phase_name}/cross_entropy_loss_episode"] = cross_entropy_loss(
        query_logits, query_targets
    )

    wandb.log(episode_metrics)

    return episode_metrics


def collect_episode_per_step_metrics(
    support_logits, support_targets, query_logits, query_targets, phase_name, task_idx,
):
    step_metrics = defaultdict(float)

    step_metrics[f"{phase_name}/task_idx={task_idx}/support_accuracy_step"] = accuracy(
        support_logits, support_targets
    )

    step_metrics[
        f"{phase_name}/task_idx={task_idx}/support_cross_entropy_loss_step"
    ] = cross_entropy_loss(support_logits, support_targets)

    step_metrics[f"{phase_name}/task_idx={task_idx}/query_accuracy_step"] = accuracy(
        query_logits, query_targets
    )

    step_metrics[
        f"{phase_name}/task_idx={task_idx}/query_cross_entropy_loss_step"
    ] = cross_entropy_loss(query_logits, query_targets)

    wandb.log(step_metrics)


class FSmethod(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(
        self, args: argparse.Namespace, logger: Optional[object] = None
    ) -> None:
        super(FSmethod, self).__init__()

    def forward(
        self,
        model: torch.nn.Module,
        support: Tensor,
        query: Tensor,
        y_s: Tensor,
        y_q: Tensor,
        task_ids: Tuple[int, int] = None,
        phase_name: str = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        args:
            model: Network to train/test with
            support: Tensor representing support images, of shape [n_tasks, n_support, C, H, W]
                     where n_tasks is the batch dimension (only useful for fixed-dimension tasks)
                     and n_support the total number of support samples
            query: Tensor representing query images, of shape [n_tasks, n_query, C, H, W]
                     where n_tasks is the batch dimension (only useful for fixed-dimension tasks)
                     and n_query the total number of query samples
            y_s: Tensor representing the support labels of shape [n_tasks, n_support]
            y_q: Tensor representing the query labels of shape [n_tasks, n_query]
            metrics: A dictionnary of Metric objects to be filled during inference
                    (mostly useful if the method performs test-time inference). Refer to tim.py for
                    an instance of usage
            task_ids: Start and end tasks ids. Only used to fill the metrics dictionnary.

        returns:
            loss: Tensor of shape [] representing the loss to be minimized (for methods using episodic training)
            soft_preds: Tensor of shape [n_tasks, n_query, K], where K is the number of classes in the task,
                        representing the soft predictions of the method for the input query samples.
        """
        raise NotImplementedError
