import argparse
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .method import (
    FewShotMethod,
    collect_episode_per_step_metrics,
    collect_episode_metrics,
)
from .utils import get_one_hot, extract_features


class Finetune(FewShotMethod):
    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    """

    def __init__(self, args: argparse.Namespace):
        self.temp = args.temp
        self.iter = args.iter
        self.finetune_all_layers = args.finetune_all_layers
        self.cosine_head = args.finetune_cosine_head
        self.weight_norm = args.finetune_weight_norm
        self.lr = args.finetune_lr

        if not hasattr(self, "episode_metrics"):
            self.episode_metrics = defaultdict(list)

        self.episodic_training = False
        self.weight_decay = args.finetune_weight_decay

        super().__init__(args)

    def _do_data_dependent_init(self, classifier: nn.Module, feat_s: Tensor):
        """Returns ops for the data-dependent init of g and maybe b_fc."""
        w_fc_normalized = F.normalize(classifier.weight_v, dim=1)  # [num_classes, d]
        output_init = feat_s @ w_fc_normalized.t()  # [n_s, num_classes]
        var_init = output_init.var(0, keepdim=True)  # [num_classes]
        # Data-dependent init values.
        classifier.weight_g.data = 1.0 / torch.sqrt(var_init + 1e-10)

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
        Corresponds to the TIM-GD inference
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot]


        updates :
            self.weights : tensor of shape [n_task, num_class, feature_dim]
            :param model:
            :param support:
            :param query:
            :param y_s:
            :param y_q:
            :param task_ids:
            :param phase_name:
            :return:
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        num_classes = y_s.unique().size(0)
        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Initialize classifier
        with torch.no_grad():
            feat_s, feat_q = extract_features(support, query, model)

            classifier = nn.Linear(feat_s.size(-1), num_classes, bias=False).to(device)
            if self.weight_norm:
                classifier = nn.utils.weight_norm(classifier, name="weight")

            # self._do_data_dependent_init(classifier, feat_s)

            if self.cosine_head:
                feat_s = F.normalize(feat_s, dim=-1)
                feat_q = F.normalize(feat_q, dim=-1)

            logits_q = self.temp * classifier(feat_q)
            logits_s = self.temp * classifier(feat_s)

        collect_episode_per_step_metrics(
            support_logits=logits_s,
            support_targets=y_s,
            query_logits=logits_q,
            query_targets=y_q,
            phase_name=phase_name,
            task_idx=task_ids,
            step_idx=0,
        )
        # Define optimizer
        if self.finetune_all_layers:
            params = list(model.parameters()) + list(classifier.parameters())
        else:
            params = list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        # Run adaptation
        for step_idx in range(1, self.iter):
            if self.finetune_all_layers:
                model.train()
                feat_s, feat_q = extract_features(support, query, model)
                if self.cosine_head:
                    feat_s = F.normalize(feat_s, dim=-1)
                    feat_q = F.normalize(feat_q, dim=-1)

            logits_s = self.temp * classifier(feat_s)
            probs_s = logits_s.softmax(-1)
            loss = -(y_s_one_hot * probs_s.log()).sum(-1).mean(-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                logits_q = self.temp * classifier(feat_q)

                collect_episode_per_step_metrics(
                    support_logits=logits_s,
                    support_targets=y_s,
                    query_logits=logits_q,
                    query_targets=y_q,
                    phase_name=phase_name,
                    task_idx=task_ids,
                    step_idx=step_idx,
                )

        return collect_episode_metrics(
            query_logits=logits_q,
            query_targets=y_q,
            phase_name=phase_name,
            step_idx=task_ids,
        )
