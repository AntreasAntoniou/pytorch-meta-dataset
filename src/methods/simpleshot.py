import argparse
from typing import Tuple
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .method import FSmethod, collect_episode_metrics
from .utils import extract_features, compute_centroids


class SimpleShot(FSmethod):
    """
    Implementation of SimpleShot method https://arxiv.org/abs/1911.04623
    """

    def __init__(self, args: argparse.Namespace):
        self.iter = args.iter
        self.episodic_training = False
        self.extract_batch_size = args.extract_batch_size

        super().__init__(args)

    def get_logits(self, samples: Tensor) -> Tensor:
        """
        inputs:
            samples : tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        return (
            samples.matmul(self.weights.transpose(1, 2))
            - 1 / 2 * (self.weights ** 2).sum(2).view(n_tasks, 1, -1)
            - 1 / 2 * (samples ** 2).sum(2).view(n_tasks, -1, 1)
        )

    def get_preds(self, samples: Tensor) -> Tensor:
        """
        inputs:
            samples : tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        return logits.argmax(2)

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

        model.eval()
        with torch.no_grad():
            feat_s, feat_q = extract_features(
                self.extract_batch_size, support, query, model
            )

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize weights
        self.weights = compute_centroids(feat_s, y_s)

        logits_q = self.get_logits(feat_q)

        return collect_episode_metrics(
            query_logits=logits_q, query_targets=y_q, phase_name=phase_name,
        )
