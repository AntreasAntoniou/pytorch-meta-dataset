import argparse
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from loguru import logger
from .utils import get_one_hot, compute_centroids, extract_features
from .method import FSmethod, collect_episode_per_step_metrics, collect_episode_metrics
from ..metrics import Metric


class ProtoNet(FSmethod):
    """
    Implementation of ProtoNet method https://arxiv.org/abs/1703.05175
    """

    def __init__(self, args: argparse.Namespace):
        self.extract_batch_size = args.extract_batch_size
        self.normamlize = False

        super().__init__(args)

    def forward(
        self,
        model: torch.nn.Module,
        support: Tensor,
        query: Tensor,
        y_s: Tensor,
        y_q: Tensor,
        task_ids: Tuple[int, int] = None,
        phase_name: str = None,
    ) -> None:
        """
        inputs:
            support : tensor of size [n_task, s_shot, c, h, w]
            query : tensor of size [n_task, q_shot, c, h, w]
            y_s : tensor of size [n_task, s_shot]
            y_q : tensor of size [n_task, q_shot]
        """
        num_classes = y_s.unique().size(0)

        with torch.set_grad_enabled(self.training):
            z_s, z_q = extract_features(0, support, query, model)

        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]
        l2_distance = torch.cdist(z_q, centroids) ** 2  # [batch, q_shot, num_class]

        log_probas = (-l2_distance).log_softmax(-1)  # [batch, q_shot, num_class]
        one_hot_q = get_one_hot(y_q, num_classes)  # [batch, q_shot, num_class]
        ce = -(one_hot_q * log_probas).sum(-1)  # [batch, q_shot, num_class]

        return collect_episode_metrics(
            query_logits=ce, query_targets=y_q, phase_name=phase_name,
        )
