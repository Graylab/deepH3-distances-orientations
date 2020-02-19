import argparse
import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm import tqdm
from datetime import datetime
from deeph3 import H3ResNet
from deeph3.util import time_diff, RawTextArgumentDefaultsHelpFormatter
from deeph3.data_util.H5AntibodyDataset import h5_antibody_dataloader
from deeph3.preprocess.create_antibody_db import download_chothias
from deeph3.preprocess.generate_h5_antibody_files import antibody_to_h5


_output_names = ['dist', 'omega', 'theta', 'phi']


def train(model, train_loader, validation_loader, optimizer, epochs, device,
          criterion, lr_modifier, writer, save_file, properties=None):
    """"""
    properties = {} if properties is None else properties
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)
    for epoch in range(epochs):
        train_losses = _train_epoch(model, train_loader, optimizer, device, criterion)
        total_train_loss = train_losses[-1]
        lr_modifier.step(total_train_loss)

        avg_train_losses = train_losses / len(train_loader)
        train_loss_dict = dict(zip(_output_names + ['total'], avg_train_losses.tolist()))
        writer.add_scalars('train_loss', train_loss_dict, global_step=epoch)
        print('\nAverage training loss (epoch {}): {}'.format(
            epoch, train_loss_dict))

        #validation_losses = _validate(model, train_loader, optimizer, device, criterion)
        #avg_validation_losses = validation_losses / len(validation_loader)
        #writer.add_scalars('validation_loss', dict(
        #    zip(_output_names + ['total'], avg_validation_losses)), global_step=epoch)
        #print('\nAverage validation loss (epoch {}): {}'.format(
        #    epoch, avg_validation_losses.tolist()))

    properties.update({'model_state_dict': model.state_dict()})
    torch.save(properties, save_file)


def _train_epoch(model, train_loader, optimizer, device, criterion):
    """Trains a model for one epoch"""
    model.train()
    running_losses = torch.zeros(5)
    for inputs, labels in tqdm(train_loader, total=len(train_loader)):
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
            return outputs, torch.Tensor([float(l.item()) for l in losses])

        outputs, batch_loss = handle_batch()
        running_losses += batch_loss
    return running_losses


def _validate(model, validation_loader, optimizer, device, criterion):
    """"""
    raise NotImplementedError()


def _get_args():
    """Gets command line arguments"""
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
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save model every X number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of proteins per batch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for Adam')
    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')
    train_py_path = os.path.dirname(os.path.realpath(__file__))
    default_h5_file = os.path.join(train_py_path, 'data/antibody.h5')
    parser.add_argument('--h5_file', type=str, default=default_h5_file)
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(train_py_path, 'models/model_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    return parser.parse_args()


def _check_for_h5_file(h5_file):
    """Checks for a H5 file. If unavailable, downloads/creates files from SabDab."""
    if not os.path.isfile(h5_file):
        train_py_path = os.path.dirname(os.path.realpath(__file__))
        ab_dir = os.path.join(train_py_path, 'data/antibody_database')
        print('No H5 file found at {}, creating new file in {}/ ...'.format(
            h5_file, ab_dir))
        if not os.path.isdir(ab_dir):
            print('{}/ does not exist, creating {}/ ...'.format(ab_dir, ab_dir))
            os.mkdir(ab_dir)
        pdb_files = [f.endswith('pdb') for f in os.listdir(ab_dir)]
        if len(pdb_files) == 0:
            print('No PDB files found in {}, downloading PDBs ...'.format(ab_dir))
            download_chothias()
        print('Creating new h5 file at {} using data from {}/ ...'.format(h5_file, ab_dir))
        antibody_to_h5(ab_dir, h5_file, print_progress=True)


def _cli():
    """Command line interface for train.py when it is run as a script"""
    args = _get_args()
    properties = dict(
        num_out_bins=args.num_bins, num_blocks1D=args.num_blocks1D,
        num_blocks2D=args.num_blocks2D, dropout_proportion=args.dropout,
        dilation_cycle=args.dilation_cycle)
    model = H3ResNet(21, **properties)
    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    optimizer = Adam(model.parameters(), lr=args.lr)
    properties.update({'lr': args.lr})
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    h5_file = args.h5_file
    _check_for_h5_file(h5_file)
    train_loader = h5_antibody_dataloader(filename=h5_file,
                                          num_bins=args.num_bins, batch_size=args.batch_size)
    validation_loader = None
    lr_modifier = ReduceLROnPlateau(optimizer, verbose=True)
    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        print('Making {} ...'.format(out_dir))
        os.mkdir(out_dir)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    print('Arguments:\n', args)
    print('Model:\n', model)

    train(model=model, train_loader=train_loader, validation_loader=validation_loader,
          optimizer=optimizer, device=device, epochs=args.epochs,
          criterion=criterion, lr_modifier=lr_modifier, writer=writer,
          save_file=os.path.join(out_dir, 'model.p'),
          properties=properties)


if __name__ == '__main__':
    _cli()

