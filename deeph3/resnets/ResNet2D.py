"""PyTorch implementation of a ResNet with 2D CNNs

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """A basic residual block with 2D CNNs
    Defines a convolutional ResNet block with the following architecture:

    -- Shortcut Path -->

    +-------- Shortcut Layer --------+
    |                                |
    X -> Conv2D -> Act -> Conv2D -> Sum -> Act -> Output

    -- Main Path -->

    The shortcut layer defaults to zero padding if the non-plane dimensions of
    X do not change after convolutions. Otherwise, it defaults to a 2D
    convolution to match dimensions.
    """
    expansion = 1
 
    def __init__(self, in_planes, planes, kernel_size=(3, 3), dilation=(1, 1), stride=1, shortcut=None):
        """
        :param in_planes: The number of input planes (features)
        :param planes: The number of output planes (features)
        :param kernel_size: Width of the kernel used in the Conv2D convolution.
        :param stride: Stride used in the Conv2D convolution.
        :param shortcut: Callable function to for the shortcut path
        """
        super(ResBlock2D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, dilation=dilation, stride=(stride, 1),
                               padding=(((kernel_size[0] - 1) * dilation[0])//2, ((kernel_size[0] - 1) * dilation[0])//2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = F.relu
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, dilation=dilation, stride=(stride, 1),
                               padding=(((kernel_size[0] - 1) * dilation[0])//2, ((kernel_size[0] - 1) * dilation[0])//2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        
        # Default zero padding shortcut
        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, 0, 0, planes - x.shape[1], 0, 0))
        # Default conv1D shortcut
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        # User defined shortcut
        else:
            self.shortcut = shortcut

    def forward(self, x):
        """
        :param x: A FloatTensor to propagate forward
        :type x: torch.Tensor
        :return:
        """
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, in_channels, block, num_blocks, init_planes=64, kernel_size=3, dilation_cycle=5):
        """
        :param in_channels: The number of channels coming from the input.
        :type in_channels: int
        :param block: The type of residual block to use.
        :type block: torch.nn.Module
        :param num_blocks:
            A list of the number of blocks per layer. Each layer increases the
            number of channels by a factor of 2.
        :type num_blocks: List[int]
        :param init_planes: The number of planes the first 1D CNN should output.
                            Must be a power of 2.
        :type init_planes: int
        :param kernel_size: Size of the convolving kernel used in the Conv2D
                            convolution.
        :type kernel_size: int
        """
        super(ResNet2D, self).__init__()
        # Check if the number of initial planes is a power of 2, done for faster computation on GPU
        if not (init_planes != 0 and ((init_planes & (init_planes - 1)) == 0)):
            raise ValueError('The initial number of planes must be a power of 2')

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.init_planes = init_planes
        self.in_planes = self.init_planes  # Number of input planes to the final layer
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=kernel_size,
                               stride=(1, 1), padding=(kernel_size//2, kernel_size//2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layers = []
        # Raise the number of planes by a power of two for each layer
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(block, int(self.init_planes * math.pow(2, i)),
                                         num_blocks[i], stride=1, kernel_size=kernel_size,
                                         dilation_cycle=dilation_cycle)
            self.layers.append(new_layer)

            # Done to ensure layer information prints out when print() is called
            setattr(self, 'layer{}'.format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size, dilation_cycle):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            dilation = int(math.pow(2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            layers.append(block(self.in_planes, planes, stride=stride,
                                kernel_size=kernel_size, dilation=(dilation, dilation)))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


def ResNet2D18(in_channels, **kwargs):
    return ResNet2D(in_channels, ResBlock2D, [2, 2, 2, 2], **kwargs)


def ResNet2D34(in_channels, **kwargs):
    return ResNet2D(in_channels, ResBlock2D, [3, 4, 6, 3], **kwargs)

