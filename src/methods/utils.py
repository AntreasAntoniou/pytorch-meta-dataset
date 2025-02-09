import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def compute_centroids(z_s: Tensor, y_s: Tensor):
    """
    inputs:
        z_s : torch.Tensor of size [*, s_shot, d]
        y_s : torch.Tensor of size [*, s_shot]
    updates :
        centroids : torch.Tensor of size [*, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(
        -2, -1
    )  # [*, K, s_shot]
    centroids = one_hot.matmul(z_s) / one_hot.sum(-1, keepdim=True)  # [*, K, d]

    return centroids


def get_one_hot(y_s: Tensor, num_classes: int):
    """
    args:
        y_s : torch.Tensor of shape [*]
    returns
        one_hot : torch.Tensor of shape [*, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)

    return one_hot


def extract_features(support: Tensor, query: Tensor, model: nn.Module):
    """
    Extract features from support and query set using the provided model
        args:
            x_s : torch.Tensor of size [batch, s_shot, c, h, w]
        returns
            z_s : torch.Tensor of shape [batch, s_shot, d]
            z_s : torch.Tensor of shape [batch, q_shot, d]
    """
    # Extract support and query features
    shots_s, C, H, W = support.size()
    shots_q = query.size(0)

    feat_s = model(support.view(shots_s, C, H, W), feature=True)
    feat_q = model(query.view(shots_q, C, H, W), feature=True)
    feat_s = feat_s.view(shots_s, -1)
    feat_q = feat_q.view(shots_q, -1)

    return feat_s, feat_q


def batch_feature_extract(
    model: nn.Module, t: Tensor, bs: int, device: torch.device
) -> Tensor:
    shots: int
    shots, C, H, W = t.size()

    feat: Tensor
    feats: List[Tensor] = []
    for i in range(math.ceil(shots / bs)):
        start = i * bs
        end = min(shots, (i + 1) * bs)

        x = t[0, start:end, ...]
        x = x.to(device)

        feat = model(x, feature=True)
        feats.append(feat)

    feat_res = torch.cat(feats, 0).unsqueeze(0)

    return feat_res
