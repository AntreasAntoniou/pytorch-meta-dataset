import argparse
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .method import FSmethod, collect_episode_metrics
from .utils import get_one_hot, extract_features, compute_centroids


class TIM(FSmethod):
    """Implementation of TIM method (NeurIPS 2020) https://arxiv.org/abs/2008.11297"""

    def __init__(self, args: argparse.Namespace):
        self.temp = args.temp
        self.loss_weights = args.loss_weights.copy()
        self.iter = args.iter
        self.extract_batch_size = args.extract_batch_size
        self.episodic_training = False

        self.weights: Tensor  # Will be init at the first forward
        super().__init__(args)

    def get_logits(self, samples: Tensor) -> Tensor:
        """
        inputs:
            samples : Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        return self.temp * (
            samples.matmul(self.weights.transpose(1, 2))
            - 1 / 2 * (self.weights ** 2).sum(2).view(n_tasks, 1, -1)
            - 1 / 2 * (samples ** 2).sum(2).view(n_tasks, -1, 1)
        )

    def get_preds(self, samples: Tensor) -> Tensor:
        """
        inputs:
            samples : Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        return logits.argmax(2)

    def compute_lambda(self, support: Tensor, query: Tensor, y_s: Tensor) -> None:
        """
        inputs:
            support : Tensor of shape [n_task, s_shot, feature_dim]
            query : Tensor of shape [n_task, q_shot, feature_dim]
            y_s : Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)

        if self.loss_weights[0] == "auto":
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q


class TIM_GD(TIM):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.lr = args.tim_lr

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
        See method.py for description of arguments.
        """
        model.eval()

        # Metric dic
        num_classes = y_s.unique().size(0)
        with torch.no_grad():
            feat_s, feat_q = extract_features(
                self.extract_batch_size, support, query, model
            )

        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize weights
        self.compute_lambda(support=feat_s, query=feat_q, y_s=y_s)
        self.weights = compute_centroids(feat_s, y_s)

        # Run adaptation
        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        for _ in range(self.iter):
            logits_s = self.get_logits(feat_s)
            logits_q = self.get_logits(feat_q)

            ce = -(y_s_one_hot * logits_s.log_softmax(2)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = -(q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)

            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return collect_episode_metrics(
            query_logits=logits_q, query_targets=y_q, phase_name=phase_name,
        )


class TIM_ADM(TIM):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.α = args.alpha

    def q_update(self, P: Tensor) -> None:
        """
        inputs:
            P : Tensor of shape [n_tasks, q_shot, num_class]
                where P[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        """
        β, α = self.loss_weights[1], self.loss_weights[2]
        # alpha = l2 / l3
        # beta = l1 / (l1 + l3)

        # Q = (P ** (1+α)) / ((P ** (1+α)).sum(dim=1, keepdim=True) + 1e-10) ** β
        Q = (P ** (1 + α / β)) / (
            (P ** (1 + α / β)).sum(dim=1, keepdim=True) + 1e-10
        ) ** (1 / (1 + β))

        self.Q = Q / Q.sum(dim=2, keepdim=True)

    def weights_update(
        self, support: Tensor, query: Tensor, y_s_one_hot: Tensor
    ) -> None:
        """
        Corresponds to w_k updates
        inputs:
            support : Tensor of shape [n_task, s_shot, feature_dim]
            query : Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : Tensor of shape [n_task, s_shot, num_classes]


        updates :
            self.weights : Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)

        P_s = self.get_logits(support).softmax(2)
        P_q = self.get_logits(query).softmax(2)

        src_weight = self.loss_weights[0] / (
            self.loss_weights[1] + self.loss_weights[2]
        )
        qry_weight = self.N_s / self.N_q

        src_part = src_weight * (
            y_s_one_hot.transpose(1, 2).matmul(support)
            + (
                self.weights * P_s.sum(1, keepdim=True).transpose(1, 2)
                - P_s.transpose(1, 2).matmul(support)
            )
        )
        src_norm = src_weight * y_s_one_hot.sum(1).view(n_tasks, -1, 1)  # noqa: E127

        qry_part = qry_weight * (
            self.Q.transpose(1, 2).matmul(query)
            + (
                self.weights * P_q.sum(1, keepdim=True).transpose(1, 2)
                - P_q.transpose(1, 2).matmul(query)
            )
        )
        qry_norm = qry_weight * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.α * (new_weights - self.weights)

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
        See method.py for description of arguments.
        """

        model.eval()

        # Metric dic
        num_classes = y_s.unique().size(0)
        with torch.no_grad():
            feat_s, feat_q = extract_features(
                self.extract_batch_size, support, query, model
            )

        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize weights
        t0 = time.time()
        self.compute_lambda(support=feat_s, query=feat_q, y_s=y_s)
        self.weights = compute_centroids(feat_s, y_s)

        # Run adaptation
        y_s_one_hot = get_one_hot(y_s, num_classes)
        P_q = self.get_logits(feat_q).softmax(2)
        for _ in range(1, self.iter):
            self.q_update(P=P_q)
            self.weights_update(feat_s, feat_q, y_s_one_hot)
            logits_q = self.get_logits(feat_q)
            P_q = logits_q.softmax(2)

        return collect_episode_metrics(
            query_logits=logits_q, query_targets=y_q, phase_name=phase_name,
        )
