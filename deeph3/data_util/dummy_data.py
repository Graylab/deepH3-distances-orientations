import math
import numpy as np
import os
from torchvision import transforms
from torchvision.datasets import MNIST


def load_dummy_data1D(seq_len=400, num_seqs=500):
    """Extremely simple sequence data to test models using Conv1D throughout development
    :param seq_len: Length of a sequence.
    :type seq_len: int
    :param num_seqs: The number of sequences.
    :type num_seqs: int
    :return:
    """
    funcs = [[math.sin, math.cos, lambda z: float(z % 20)],
             [math.cos, math.sin, lambda z: float(z % 20)],
             [math.sin, math.cos, lambda z: float(z % 18)]]
    X = [[[func(i) for func in funcs[j % len(funcs)]] for i in range(seq_len)] for j in range(num_seqs)]
    y = [[1 if j % len(funcs) == label else 0 for label in range(len(funcs))] for j in range(num_seqs)]
    return np.array(X), np.array(y)


def load_MNIST():
    """"""
    if not os.path.isdir('./data/'):
        raise NotADirectoryError('./data/ must be present in the executing directory.')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    mnist_train = MNIST(root='./data/', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data/', train=False, download=True, transform=transform)

    return mnist_train, mnist_test

