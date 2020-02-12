import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from torch.optim import Adam
from deeph3 import H3ResNet
from deeph3.util import time_diff, RawTextArgumentDefaultsHelpFormatter
from deeph3.data_util.H5AntibodyDataset import h5_antibody_dataloader, H5AntibodyBatch


_output_names = ['dist', 'omega', 'theta', 'phi']


def train(model, train_loader, optimizer, epochs, device, criterion):
    """"""
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)
    model.train()
    avg_losses = torch.zeros(5)
    for epoch in range(epochs):
        avg_losses += _train_batch(model, train_loader, optimizer, epoch, device, criterion)


def _train_batch(model, train_loader, optimizer, epoch, device, criterion):
    running_loss = torch.zeros(5)
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_start = time()
        inputs = inputs.to(device)
        labels = [label.to(device) for label in labels]

        optimizer.zero_grad()

        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            outputs = model(inputs).transpose(0, 1)
            losses = [criterion(output, label) for output, label
                      in zip(outputs, labels)]
            total_loss = sum(losses)
            losses.append(total_loss)

            total_loss.backward()
            optimizer.step()
            return outputs, torch.Tensor([float(loss.item()) for loss in losses])

        outputs, batch_loss = handle_batch()
        running_loss += batch_loss
        print('\nTotal time for batch of {} samples: {}'.format(
            len(labels[0]), time_diff(batch_start, time())))
    avg_loss = running_loss / len(train_loader)
    print('\nAverage loss (epoch {}): {}'.format(epoch, avg_loss))
    return avg_loss


def cli():
    """Command line interface for train.py when it is run as a script"""
    desc = (
        '''
        ''')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=RawTextArgumentDefaultsHelpFormatter)
    # Model architecture arguments
    parser.add_argument('--num_blocks1D', type=int, default=3,
                        help='Number of one-dimensional ResNet blocks to use.')
    parser.add_argument('--num_blocks2D', type=int, default=21,
                        help='Number of two-dimensional ResNet blocks to use.')
    parser.add_argument('--dilation_cycle', type=int, default=5)
    parser.add_argument('--num_bins', type=int, default=26,
                        help=('Number of bins to discretize the continuous '
                              'distance, and angle values into.\n'
                              'Example:\n'
                              'For residue pairs i and j, let d be the euclidean '
                              'distance between them in Angstroms. A num_bins of '
                              '26 would discretize each d into the following bins:\n'
                              '[d < 4.0, 4.0 <= d < 4.5, 4.5 <= d < 5.0, ..., 16 <= d < Inf]\n'
                              'Depending on the type of inter-residue angle, '
                              'angles are binned into 26 evenly spaced bins between '
                              '0 (or -180) and 180 degrees'))
    parser.add_argument('--dropout', type=float, default=0.2,
                        help=('The chance of entire channels being zeroed out '
                              'during training'))

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of proteins per batch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for Adam')

    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')

    train_py_path = os.path.dirname(os.path.realpath(__file__))
    default_training_file = os.path.join(train_py_path, 'data/antibody.h5')
    parser.add_argument('--training_file', type=str, default=default_training_file)

    args = parser.parse_args()
    model = H3ResNet(21, num_out_bins=args.num_bins, num_blocks1D=args.num_blocks1D,
                     num_blocks2D=args.num_blocks2D, dropout_proportion=args.dropout,
                     dilation_cycle=args.dilation_cycle)
    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    data_loader = h5_antibody_dataloader(filename=args.training_file,
                                         num_bins=args.num_bins, batch_size=args.batch_size)

    train(model=model, train_loader=data_loader, optimizer=optimizer, device=device,
          epochs=args.epochs, criterion=criterion)


if __name__ == '__main__':
    cli()

