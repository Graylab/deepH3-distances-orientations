import torch.nn as nn


class Flatten(nn.Module):
    """Flattens an nd tensor to a 1d tensor"""
    def forward(self, x):
        return x.view(x.size()[0], -1)

