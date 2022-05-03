import torch
import torch.nn.functional as F


def accuracy(logits, targets):
    return (logits.argmax(-1) == targets).float().mean()


def cross_entropy_loss(logits, targets):
    return F.cross_entropy(input=logits, target=targets)
