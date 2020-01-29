from deeph3.cli.trainer_cli import get_cli_input


desc = ('Trains a ResNet using H5 files made of antibody data\n'
        'Default architecture:\n'
        '3 one-dimensional ResNet Blocks\n'
        '10 two-dimensional ResNet Blocks\n'
        'SGD optimizer with 0.01 learning rate, and 0.5 momentum')
get_cli_input(desc, 
              default_training_file='antibody_train.h5',
              default_validation_file='antibody_validation.h5', 
              dataset_type='ab', 
              in_planes=21)

