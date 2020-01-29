import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input_, target, input_weights=None):
        logs = F.log_softmax(input_, 1)
        if input_weights is not None:
            logs *= input_weights
        return F.nll_loss(logs, target, ignore_index=self.ignore_index,
                          reduction=self.reduction, weight=self.weight)

