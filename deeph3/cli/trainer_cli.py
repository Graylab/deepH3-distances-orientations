import argparse
import sys
from os.path import join, isabs
from deeph3.trainer import h5_trainer


def get_cli_input(desc, default_training_file, default_validation_file, dataset_type=None, in_planes=20):
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('data_dir', type=str,
                        help='The directory containing the training and validation '
                             'h5 files.')
    if dataset_type is None:
        parser.add_argument('--dataset_type', type=str, default='ab',
                            help='The type of protein data in the h5 file. '
                                 'Either \'ab\' (antibody) or \'pnet\' ('
                                 'ProteinNet)')
    # File arguments
    parser.add_argument('--training_file', type=str, default=default_training_file)
    parser.add_argument('--validation_file', type=str, default=default_validation_file)
    parser.add_argument('--fasta_file', type=str, default=None)

    # Model architecture arguments
    parser.add_argument('--num_blocks1D', type=int, default=3)
    parser.add_argument('--num_blocks2D', type=int, default=10)
    parser.add_argument('--dilation_cycle', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs to run for.')
    parser.add_argument('--max_training_seq_len', type=int, default=1000,
                        help='Maximum sequence length of any protein in the '
                             'training set')
    parser.add_argument('--max_validation_seq_len', type=int, default=None,
                        help='Maximum sequence length of any protein in the '
                             'validation set')

    # Iteration arguments
    parser.add_argument('--max_training_iter', type=int, default=None)
    parser.add_argument('--max_validation_iter', type=int, default=None)

    # Output arguments
    parser.add_argument('--print_file', type=str, default=None,
                        help='File to print to. If None, the console is used.')
    parser.add_argument('--models_dir', type=str, default=None,
                        help='The directory to save models to.')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save every X batches, per epoch.')

    # Loss/Criterion arguments
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss',
                        help='The loss function to use. Can be '
                             'CrossEntropyLoss or WeightedCrossEntropyLoss')

    # Learning rate scheduler arguements
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                        help='The learning rate scheduler to use')
    parser.add_argument('--factor', type=float, default=0.01,
                        help='The factor by which the learning rate is reduced')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='The factor by which the learning rate is reduced')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='A lower bound on the learning rate.')
    parser.add_argument('--patience', type=int, default=1,
                        help='Number of epochs with no improvement after which '
                             'learning rate will be reduced. For example, if '
                             'patience = 2, then we will ignore the first 2 '
                             'epochs with no improvement, and will only '
                             'decrease the LR after the 3rd epoch if the '
                             'loss still hasn’t improved then.')

    # Optimizer type argument
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='The optimizer being used. Can be SGD, or '
                             'AdaBound')

    parser.add_argument('--L2', type=float, default=0.0,
                        help='The L2 penalty applied to the optimizer')
    # kwargs for SGD
    parser.add_argument('--lr', type=float, default=0.01,
                        help='If SGD is being used, is the learning rate of the '
                             'model.')
    parser.add_argument('--momentum', type=float, default=0.90,
                        help='If SGD is being used, is the momentum of the model')
    # kwargs for AdaBound
    parser.add_argument('--init_lr', type=float, default=0.001,
                        help='If AdaBound is being used, is the initial '
                             'learning rate of the model.')
    parser.add_argument('--final_lr', type=float, default=0.1,
                        help='If AdaBound is being used, is the final '
                             'learning rate of the model.')

    args = parser.parse_args()

    if args.print_file is not None:
        sys.stdout = open(args.print_file, 'w')

    # Data variables
    data_dir = args.data_dir
    train_file = args.training_file
    validation_file = args.validation_file
    fasta_file = args.fasta_file

    # Create paths to files
    train_h3_file_path = str(join(data_dir, train_file))
    validation_h3_file_path = str(join(data_dir, validation_file))

    if fasta_file is not None:
        if isabs(fasta_file):
            fasta_file_path = fasta_file
        else:
            fasta_file_path = str(join(data_dir, fasta_file))
    else:
        fasta_file_path = None

    # Instantiate parameters
    num_blocks1D = args.num_blocks1D
    num_blocks2D = args.num_blocks2D
    dilation_cycle = args.dilation_cycle
    batch_size = args.batch_size
    save_every = args.save_every
    max_training_seq_len = args.max_training_seq_len
    max_validation_seq_len = args.max_validation_seq_len
    max_training_iter = args.max_training_iter
    max_validation_iter = args.max_validation_iter
    models_dir = args.models_dir
    epochs = args.epochs

    # Create optimizer parameters
    optimizer_type = args.optimizer
    if optimizer_type == 'SGD':
        lr = args.lr
        momentum = args.momentum
        optimizer_kwargs = dict(lr=lr, momentum=momentum)
    elif optimizer_type == 'AdaBound':
        lr = args.init_lr
        final_lr = args.final_lr
        optimizer_kwargs = dict(lr=lr, final_lr=final_lr)
    elif optimizer_type == 'Adam':
        lr = args.init_lr
        optimizer_kwargs = dict(lr=lr)
    else:
        raise ValueError('optimizer must be SGD or AdaBound')
    weight_decay = args.L2
    optimizer_kwargs.update(dict(weight_decay=weight_decay))

    lr_scheduler_type = args.lr_scheduler
    factor = args.factor
    threshold = args.threshold
    min_lr = args.min_lr
    patience = args.patience
    lr_scheduler_kwargs = dict(factor=factor, min_lr=min_lr, patience=patience,
                               threshold=threshold)

    h5_trainer(in_planes=in_planes,
               train_file_path=train_h3_file_path,
               validation_file_path=validation_h3_file_path,
               num_blocks1D=num_blocks1D,
               num_blocks2D=num_blocks2D,
               dilation_cycle=dilation_cycle,
               optimizer_type=optimizer_type,
               optimizer_kwargs=optimizer_kwargs,
               lr_modifer_type=lr_scheduler_type,
               lr_modifier_kwargs=lr_scheduler_kwargs,
               dataset_type=dataset_type,
               batch_size=batch_size,
               save_every=save_every,
               max_training_seq_len=max_training_seq_len,
               max_validation_seq_len=max_validation_seq_len,
               max_training_iter=max_training_iter,
               max_validation_iter=max_validation_iter,
               out_fasta=fasta_file_path,
               models_dir=models_dir,
               epochs=epochs)


if __name__ == '__main__':
    def main():
        desc = ('Trains a ResNet, use pnet_trainer.py or antibody_trainer.py.\n'
                'Default architecture:\n'
                '3 one-dimensional ResNet Blocks\n'
                '10 two-dimensional ResNet Blocks\n'
                'SGD optimizer with 0.01 learning rate, and 0.5 momentum')
        get_cli_input(desc,
                      default_training_file='training_30.h5', 
                      default_validation_file='validation.h5')
    main()

