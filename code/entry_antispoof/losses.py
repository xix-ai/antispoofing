import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class DistributionLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=False, ignore_index=-100, reduce=False):
        super(DistributionLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        probs = nn.functional.log_softmax(input, dim=1)
        return -1.0 * (target * probs).mean()


class FocalAge(_WeightedLoss):
    def __init__(self, weight=None, size_average=False, ignore_index=-100, reduce=False):
        super(DistributionLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        probs = nn.functional.log_softmax(input, dim=1)

        return -(target * probs).mean()  # .sum()/len(target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce)
        loss = self.alpha * (1 - p) ** self.gamma * ce

        if self.reduce:
            return torch.mean(loss)
        return loss


class ReducedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, reduce_th=0.5, **kwargs):
        super(ReducedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.reduce_th = reduce_th

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce)
        focal_reduction = ((1.0 - p) / self.reduce_th) ** self.gamma
        focal_reduction[p < self.reduce_th] = 1
        loss = focal_reduction * ce

        if self.reduce:
            return torch.mean(loss)
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, logits=True, alpha=1, gamma=2, reduce=True, **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.logits = logits
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction="none")

        p = torch.exp(-bce)
        loss = self.alpha * (1 - p) ** self.gamma * bce

        if self.reduce:
            return torch.mean(loss)
        return loss


class CELoss(nn.Module):
    def __init__(self, reduce=True, **kwargs):
        super(CELoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, target):
        ce = F.cross_entropy(inputs, target, reduction="none")

        if self.reduce:
            return torch.mean(ce)
        return ce


class BCELoss(nn.Module):
    def __init__(self, logits=True, reduce=True, **kwargs):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction="none")

        if self.reduce:
            return torch.mean(bce)
        return bce


class DiceLoss(nn.Module):
    def __init__(self, beta=1, eps=1e-7, **kwargs):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)

        intersection = torch.sum(inputs * target)
        union = torch.sum(inputs + target)
        score = 2 * intersection / (union + self.eps)

        return 1.0 - score


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, reduce=True):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.reduce = reduce

    def forward(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=-1)

        nll_loss = -logprobs * targets
        nll_loss = torch.sum(nll_loss, dim=-1)

        smooth_loss = -torch.mean(logprobs, dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduce:
            loss = torch.mean(loss)

        return loss


def focal(*argv, **kwargs):
    return FocalLoss(*argv, **kwargs)


def rfocal(*argv, **kwargs):
    return ReducedFocalLoss(*argv, **kwargs)


def binary_focal(*argv, **kwargs):
    return BinaryFocalLoss(*argv, **kwargs)


def ce(*argv, **kwargs):
    return CELoss(*argv, **kwargs)


def binary_ce(*argv, **kwargs):
    return BCELoss(*argv, **kwargs)


def dice(*argv, **kwargs):
    return DiceLoss(*argv, **kwargs)


def label_smoothing(*argv, **kwargs):
    return LabelSmoothing(*argv, **kwargs)


def gaus_age(*argv, **kwargs):
    return DistributionLoss(*argv, **kwargs)
