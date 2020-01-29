"""PyTorch implementation of a Pre-Activation ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock1D(nn.Module):
    """Pre-activation residual block
    Defines a convolutional ResNet block with the following architecture:

    -- Shortcut Path -->

    +------------------ Shortcut Layer ----------------+
    |                                                  |
    X -> BN -> Act -> Conv1D -> BN -> Act -> Conv1D-> Sum -> Output

    -- Main Path -->

    The shortcut layer defaults to zero padding if the non-plane dimensions of
    X do not change after convolutions. Otherwise, it defaults to a 1D
    convolution, with the BN -> ReLU -> Conv1D architecture, to match dimensions.
    """
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, shortcut=None):
        """
        :param in_planes: The number of input planes (features) per timestep in
                          the sequence.
        :type in_planes: int
        :param planes: The number of planes (features) to output per timestep.
        :type planes: int
        :param kernel_size: Size of the convolving kernel used in the Conv1D
                            convolution.
        :type kernel_size: int
        :param stride: Stride used in the Conv1D convolution.
        :type stride: int
        :param shortcut:
            Callable function to be used in the shortcut path. If None, Defaults
            to zero padding if stride is one. If stride is not one, defaults to
            a 1D convolution.
        """
        super(PreActBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False,
                               padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(planes)
        self.activation = F.leaky_relu
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, bias=False,
                               padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride

        # Default zero padding shortcut
        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, planes - x.shape[1], 0, 0))
        # Default conv1D shortcut
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                self.bn1,
                self.activation,
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes))
        # User defined shortcut
        else:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out += self.shortcut(x)
        return out


class PreActBottleneck1D(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck1D, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(self.bn2(out)))
        out = self.conv3(F.leaky_relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet1D(nn.Module):
    def __init__(self, in_channels, block, num_blocks, init_planes=64, kernel_size=3):
        """
        :param in_channels: The number of channels coming from the input.
        :type in_channels: int
        :param block: The type of residual block to use.
        :type block: torch.nn.Module
        :param num_blocks: A list of the number of blocks per layer. Each layer increases the number of channels by a
                           factor of 2.
        :type num_blocks: List[int]
        :param init_planes: The number of planes the first 1D CNN should output. Must be a power of 2.
        :type init_planes: int
        :param kernel_size: Size of the convolving kernel used in the Conv1D convolution.
        :type kernel_size: int
        """
        super(PreActResNet1D, self).__init__()
        # Check if the number of initial planes is a power of 2, done for faster computation on GPU
        if not (init_planes != 0 and ((init_planes & (init_planes - 1)) == 0)):
            raise ValueError('The initial number of planes must be a power of 2')

        self.init_planes = init_planes
        self.in_planes = self.init_planes  # Number of input planes to the final layer
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv1d(in_channels, self.in_planes, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)

        self.layers = []
        # Raise the number of planes by a power of two for each layer
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(block, int(self.init_planes * math.pow(2, i)), num_blocks[0], stride=1)
            self.layers.append(new_layer)

            # Done to ensure layer information prints out when print() is called
            setattr(self, 'layer{}'.format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for layer in self.layers:
            out = layer(out)
        return out


def PreActResNet1D18(in_channels, **kwargs):
    return PreActResNet1D(in_channels, PreActBlock1D, [2, 2, 2, 2], **kwargs)


def PreActResNet1D34(in_channels, **kwargs):
    return PreActResNet1D(in_channels, PreActBlock1D, [3, 4, 6, 3], **kwargs)

