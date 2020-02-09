from deeph3 import H3ResNet
import argparse


def train(model):
    """"""
    pass


def cli():
    """Command line interface for train.py when it is run as a script"""
    desc = (
        '''
        ''')
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    # Model architecture arguments
    parser.add_argument('--num_blocks1D', type=int, default=3,
                        help='Number of one-dimensional ResNet blocks to use.')
    parser.add_argument('--num_blocks2D', type=int, default=21,
                        help='Number of two-dimensional ResNet blocks to use.')
    parser.add_argument('--dilation_cycle', type=int, default=5)
    parser.add_argument('--num_bins', type=int, default=21,
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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    model = H3ResNet(21, num_out_bins=args.num_bins, num_blocks1D=args.num_blocks1D,
                     num_blocks2D=args.num_blocks2D, dropout_proportion=args.dropout,
                     dilation_cycle=args.dilation_cycle)
    print(model)


if __name__ == '__main__':
    cli()

