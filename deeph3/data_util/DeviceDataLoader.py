import torch


class DeviceDataLoader:
    """A wrapper for DataLoaders to automatically load data to the GPU, if
    available.
    This class and its helper functions were taken from
    https://jvn.io/aakashns/fdaae0bf32cf4917a931ac415a5c31b0
    """
    def __init__(self, dl, device=None):
        """
        :param dl: The dataloader to wrap around
        :param device: The device to send data to. If None, get_default_device()
                       is used.
        """
        self.dl = dl
        self.device = device
        if device is None:
            self.device = get_default_device()
        print('Using {} as device'.format(str(self.device).upper()))

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

