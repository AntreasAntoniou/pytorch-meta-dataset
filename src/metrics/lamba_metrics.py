import torch
import torch.nn.functional as F
from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.mean(targets.eq(logits.argmax(-1)).float())


def cross_entropy_loss(logits, targets):
    return F.cross_entropy(input=logits, target=targets)
