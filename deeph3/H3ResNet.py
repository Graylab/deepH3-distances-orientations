import math
from torch import stack
import torch.nn as nn
from deeph3.resnets import ResNet1D, ResBlock1D, ResNet2D, ResBlock2D
from deeph3.layers import OuterConcatenation2D


class H3ResNet(nn.Module):
    def __init__(self, in_planes, num_out_bins=25, num_blocks1D=3, num_blocks2D=10,
                 dilation_cycle=0):
        super(H3ResNet, self).__init__()
        if isinstance(num_blocks1D, list):
            if len(num_blocks1D) > 1:
                raise NotImplementedError('Multi-layer resnets not supported')
            num_blocks1D = num_blocks1D[0]

        if isinstance(num_blocks2D, list):
            if len(num_blocks2D) > 1:
                raise NotImplementedError('Multi-layer resnets not supported')
            num_blocks2D = num_blocks2D[0]

        self._num_out_bins = num_out_bins

        self.resnet1D = ResNet1D(in_planes, ResBlock1D, [num_blocks1D],
                                 init_planes=32, kernel_size=17)
        self.seq2pairwise = OuterConcatenation2D()

        # Calculate the number of planes output from the seq2pairwise layer
        expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
        out_planes1D = self.resnet1D.init_planes * expansion1D
        in_planes2D = 2 * out_planes1D

        self.resnet2D = ResNet2D(in_planes2D, ResBlock2D, [num_blocks2D],
                                 init_planes=64, kernel_size=5, dilation_cycle=dilation_cycle)

        # Calculate the number of planes output from the ResNet2D layer
        expansion2D = int(math.pow(2, self.resnet2D.num_layers - 1))
        out_planes2D = self.resnet2D.init_planes * expansion2D

        self.out_dropout = nn.Dropout2d(p=0.2)

        # Output convolution to reduce/expand to the number of bins
        self.out_conv_dist = nn.Conv2d(out_planes2D, num_out_bins,
                                       kernel_size=self.resnet2D.kernel_size,
                                       padding=self.resnet2D.kernel_size // 2)
        self.out_conv_omega = nn.Conv2d(out_planes2D, num_out_bins,
                                        kernel_size=self.resnet2D.kernel_size,
                                        padding=self.resnet2D.kernel_size // 2)
        self.out_conv_theta = nn.Conv2d(out_planes2D, num_out_bins,
                                        kernel_size=self.resnet2D.kernel_size,
                                        padding=self.resnet2D.kernel_size // 2)
        self.out_conv_phi = nn.Conv2d(out_planes2D, num_out_bins,
                                      kernel_size=self.resnet2D.kernel_size,
                                      padding=self.resnet2D.kernel_size // 2)

    def forward(self, x):
        out = self.resnet1D(x)
        out = self.seq2pairwise(out)
        out = self.resnet2D(out)
        out = self.out_dropout(out)

        out_dist = self.out_conv_dist(out)
        out_omega = self.out_conv_omega(out)
        out_theta = self.out_conv_theta(out)
        out_phi = self.out_conv_phi(out)

        out_dist = out_dist + out_dist.transpose(2, 3)
        out_omega = out_omega + out_omega.transpose(2, 3)

        return stack([out_dist, out_omega, out_theta, out_phi]).transpose(0, 1)

