import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deeph3 import H3ResNet, WeightedCrossEntropyLoss
from deeph3.data_util import h5_proteinnet_dataloader, h5_antibody_dataloader, get_default_device, to_device
from deeph3.resnets import ResBlock2D, ResBlock1D
from adabound import AdaBound
from os import mkdir
from os.path import dirname, join, isdir, isfile, basename, split
import time


OPTMIZERS = {'SGD': SGD, 'AdaBound': AdaBound, 'Adam': Adam}
LR_MODIFIERS = {'ReduceLROnPlateau': ReduceLROnPlateau}
DATA_LOADER_MAKERS = {'ab': h5_antibody_dataloader,
                      'pnet': h5_proteinnet_dataloader}
CRITERIONS = {'CrossEntropyLoss': nn.CrossEntropyLoss,
              'WeightedCrossEntropyLoss': WeightedCrossEntropyLoss}


def get_lr_modifier_name(lr_modifier):
    for name, modifier_type in LR_MODIFIERS.items():
        if isinstance(lr_modifier, modifier_type):
            return name
    raise KeyError('{} is not a supported lr modifier'.format(lr_modifier))


def get_criterion_name(criterion):
    for name, criterion_type in CRITERIONS.items():
        if isinstance(criterion, criterion_type):
            return name

    raise KeyError('{} is not a supported criterion/loss'.format(criterion))


def get_optimizer_name(optimizer):
    for name, optim_type in OPTMIZERS.items():
        if isinstance(optimizer, optim_type):
            return name
    raise KeyError('{} is not a supported optimizer'.format(optimizer))


def load_model(file_name):
    """Loads an ab or pnet model from a correctly named file"""
    if not isfile(file_name):
        raise FileNotFoundError(f'No file at {file_name}')
    checkpoint_dict = torch.load(file_name, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    dilation_cycle = 0 if not 'dilation_cycle' in checkpoint_dict else checkpoint_dict[
        'dilation_cycle']

    in_layer = list(model_state.keys())[0]
    out_layer = list(model_state.keys())[-1]
    num_out_bins = model_state[out_layer].shape[0]
    in_planes = model_state[in_layer].shape[1]

    properties = parse_file_name(file_name)
    num_blocks1D, num_blocks2D = properties[:2]

    resnet = H3ResNet(in_planes=in_planes, num_out_bins=num_out_bins,
                      num_blocks1D=num_blocks1D, num_blocks2D=num_blocks2D,
                      dilation_cycle=dilation_cycle)
    model = nn.Sequential(resnet)
    model.load_state_dict(model_state)
    model.eval()

    return model


def file_load_checkpoint_dict(file_name):
    """Loads the model checkpoint dictionary from its file name"""
    raise NotImplementedError('Updates needed from OPTIMIZER dict creation')
    if not isfile(file_name):
        raise FileNotFoundError(f'No file at {file_name}')
    checkpoint_dict = torch.load(file_name)
    model_state = checkpoint_dict['model_state_dict']

    in_layer = list(model_state.keys())[0]
    out_layer = list(model_state.keys())[-1]
    num_out_bins = model_state[out_layer].shape[0]
    in_planes = model_state[in_layer].shape[1]

    num_blocks1D, num_blocks2D = parse_file_name(file_name)

    resnet = H3ResNet(in_planes=in_planes, num_out_bins=num_out_bins,
                      num_blocks1D=num_blocks1D, num_blocks2D=num_blocks2D)
    model = nn.Sequential(resnet)
    model.load_state_dict(model_state)
    model.train()

    #optimizer = AdaBound(model.parameters(), lr=lr, final_lr=final_lr)
    # optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    batch_loader = checkpoint_dict['batch_loader']
    epoch = checkpoint_dict['epoch']
    batch = checkpoint_dict['batch']
    criterion = checkpoint_dict['criterion']
    validation_losses = checkpoint_dict['validation_losses']
    training_losses = checkpoint_dict['training_losses']

    return dict(batch=batch, batch_loader=batch_loader, model=model,
                criterion=criterion, optimizer=optimizer, epoch=epoch,
                validation_losses=validation_losses, training_losses=training_losses)


def load_checkpoint_dict(
        in_planes, num_blocks1D, num_blocks2D, dilation_cycle,
        train_h5_file_path, validation_h5_file_path,
        batch_size=8, num_out_bins=26, dataset_type='ab',
        optimizer_type='SGD', optimizer_kwargs=None,
        criterion_type='CrossEntropyLoss', class_weights=None,
        max_training_seq_len=None, max_validation_seq_len=None,
        models_dir=None, device=None, seed=1234,
        lr_modifier_type='ReduceLROnPlateau', lr_modifier_kwargs=None):
    """Loads H3ResNet model components from memory. If unavailable, creates them

    Gets an H3ResNet model, its AntibodyBatchLoader, criterion, and optimizer
    from memory, if available. Otherwise, this will generate a new instance of
    each of these components.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if lr_modifier_kwargs is None:
        lr_modifier_kwargs = {}
    if device is None:
        device = get_default_device()

    print('Preparing data...')
    checkpoint = None
    name = '{}{}_{}_batchsize{}'.format(
        dataset_type,
        basename(train_h5_file_path).split('.')[0],
        basename(validation_h5_file_path).split('.')[0],
        batch_size)

    file_name = _get_file_name(num_blocks1D, num_blocks2D,
                               dilation_cycle, num_out_bins,
                               seed=seed,
                               name=name, models_dir=models_dir,
                               optimizer_name=optimizer_type,
                               optimizer_kwargs=optimizer_kwargs,
                               lr_modifier_name=lr_modifier_type,
                               lr_modifier_kwargs=lr_modifier_kwargs,
                               criterion_name=criterion_type)

    data_loader_maker = DATA_LOADER_MAKERS[dataset_type]
    train_loader = data_loader_maker(train_h5_file_path,
                                     batch_size=batch_size,
                                     max_seq_len=max_training_seq_len)
    validation_loader = data_loader_maker(validation_h5_file_path,
                                          batch_size=batch_size,
                                          max_seq_len=max_validation_seq_len)

    # Load from previous checkpoint, if available
    if isfile(file_name):
        checkpoint = torch.load(file_name)
        torch.manual_seed(checkpoint['seed'])
        batch = checkpoint['batch']
        batch_size = checkpoint['batch_size']
        max_validation_seq_len = checkpoint['max_validation_seq_len']
        max_training_seq_len = checkpoint['max_training_seq_len']
    else:
        torch.manual_seed(seed)
        batch = 0

    print('Creating Model...')
    resnet = H3ResNet(in_planes=in_planes, num_out_bins=num_out_bins,
                      num_blocks1D=num_blocks1D, num_blocks2D=num_blocks2D,
                      dilation_cycle=dilation_cycle)
    if class_weights is None:
        class_weights = train_loader.dataset.balanced_class_weights()
        class_weights = to_device(torch.Tensor(class_weights), device)
    model = to_device(nn.Sequential(resnet), device)
    criterion_dist = CRITERIONS[criterion_type](weight=class_weights[0],
                                                ignore_index=-1)
    criterion_omega = CRITERIONS[criterion_type](weight=class_weights[1],
                                                 ignore_index=-1)
    criterion_theta = CRITERIONS[criterion_type](weight=class_weights[2],
                                                 ignore_index=-1)
    criterion_phi = CRITERIONS[criterion_type](weight=class_weights[3],
                                               ignore_index=-1)
    criterion = [criterion_dist, criterion_omega,
                 criterion_theta, criterion_phi]
    optimizer = OPTMIZERS[optimizer_type](
        model.parameters(), **optimizer_kwargs)
    lr_modifier = LR_MODIFIERS[lr_modifier_type](
        optimizer, verbose=True, **lr_modifier_kwargs)
    epoch = 0
    validation_losses = []
    training_losses = []
    training_metrics = []
    validation_metrics = []

    # Load from last checkpoint, if available
    if checkpoint is not None:
        print('Loading from previous state ({})...'.format(file_name))
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()  # Set batch normalization layers to train mode
        criterion = checkpoint['criterion']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        validation_losses = checkpoint['validation_losses']
        training_losses = checkpoint['training_losses']
        training_metrics = checkpoint['training_metrics']
        validation_metrics = checkpoint['validation_metrics']
        seed = checkpoint['seed']
        lr_modifier.load_state_dict(checkpoint['lr_modifier_state_dict'])
    else:
        print('No previous state found at {}! '.format(file_name))
        print('Starting from random...')

    return dict(batch=batch,
                train_loader=train_loader,
                validation_loader=validation_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr_modifier=lr_modifier,
                epoch=epoch,
                seed=seed,
                dilation_cycle=dilation_cycle,
                validation_losses=validation_losses,
                training_losses=training_losses,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                batch_size=batch_size,
                max_validation_seq_len=max_validation_seq_len,
                max_training_seq_len=max_training_seq_len)


def save_train_checkpoint(dataset_type, train_loader, validation_loader,
                          model, dilation_cycle, criterion, optimizer,
                          optimizer_kwargs, epoch, batch, training_losses,
                          validation_losses, batch_size, seed,
                          max_validation_seq_len, max_training_seq_len,
                          lr_modifier, lr_modifier_kwargs,
                          training_metrics, validation_metrics, models_dir=None):
    # Overwrite the model's previous saved state
    name = '{}{}_{}_batchsize{}'.format(
        dataset_type,
        basename(train_loader.dataset.filename).split('.')[0],
        basename(validation_loader.dataset.filename).split('.')[0],
        batch_size)
    optimizer_name = get_optimizer_name(optimizer)
    lr_modifier_name = get_lr_modifier_name(lr_modifier)
    file_name = _get_model_file_name(
        model=model, dilation_cycle=dilation_cycle, criterion=criterion, optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs, seed=seed, name=name,
        lr_modifier_name=lr_modifier_name, lr_modifier_kwargs=lr_modifier_kwargs,
        models_dir=models_dir)

    properties = {'batch': batch,
                  'epoch': epoch,
                  'dilation_cycle': dilation_cycle,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_modifier_state_dict': lr_modifier.state_dict(),
                  'criterion': criterion,
                  'training_losses': training_losses,
                  'validation_losses': validation_losses,
                  'batch_size': batch_size,
                  'seed': seed,
                  'max_validation_seq_len': max_validation_seq_len,
                  'max_training_seq_len': max_training_seq_len,
                  'training_metrics': training_metrics,
                  'validation_metrics': validation_metrics}

    print('Saving model to {}...'.format(file_name))
    torch.save(properties, file_name)

    # Save the current state at the current epoch in a seperate file
    file_name = join(split(file_name)[0], '{}_epoch{}_batch{}.p'.format(
        split(file_name)[1].split('.')[0], epoch, batch))
    while True:
        try:
            torch.save(properties, file_name)
            break
        except RuntimeError:
            time.sleep(5)


def _get_model_file_name(model, dilation_cycle, criterion, optimizer_name,
                         optimizer_kwargs, lr_modifier_name, lr_modifier_kwargs,
                         seed, name, models_dir=None):
    """Gets the file name of a model given its state"""

    if isinstance(criterion, list):
        criterion = criterion[0]

    num_blocks1D = len([_ for _ in str(model).split('\n')
                        if ResBlock1D.__name__ in _])
    num_blocks2D = len([_ for _ in str(model).split('\n')
                        if ResBlock2D.__name__ in _])
    criterion_name = get_criterion_name(criterion)

    return _get_file_name(
        num_blocks1D=num_blocks1D, num_blocks2D=num_blocks2D, dilation_cycle=dilation_cycle,
        bins=criterion.weight.shape[0], optimizer_kwargs=optimizer_kwargs,
        seed=seed, name=name, models_dir=models_dir, optimizer_name=optimizer_name,
        criterion_name=criterion_name, lr_modifier_name=lr_modifier_name,
        lr_modifier_kwargs=lr_modifier_kwargs)


def _get_file_name(num_blocks1D, num_blocks2D, dilation_cycle, bins, criterion_name,
                   optimizer_name, optimizer_kwargs, lr_modifier_name,
                   lr_modifier_kwargs, seed, name, models_dir=None):
    if models_dir is None:
        models_dir = join(dirname(__file__), 'models')
    if not isdir(models_dir):
        mkdir(models_dir)
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if lr_modifier_kwargs is None:
        lr_modifier_kwargs = {}
    optimizer_str = [str(k) + '' + str(v).replace('.', 'p')
                     for k, v in optimizer_kwargs.items()]
    optimizer_str = '_'.join(optimizer_str)
    lr_modifier_str = [str(k) + '' + str(v).replace('.', 'p')
                       for k, v in lr_modifier_kwargs.items()]
    lr_modifier_str = '_'.join(lr_modifier_str)

    if isinstance(num_blocks1D, int):
        num_blocks1D = [num_blocks1D]
    if isinstance(num_blocks2D, int):
        num_blocks2D = [num_blocks2D]

    # Concatenate the number of blocks per layer with an '_'
    num_blocks1D_str = '_'.join([str(_) for _ in num_blocks1D])
    num_blocks2D_str = '_'.join([str(_) for _ in num_blocks2D])

    return join(models_dir,
                '{}model_1D{}_2D{}_bins{}_dil{}_{}_{}_{}_{}_{}_seed{}.p'.format(
                    name, num_blocks1D_str, num_blocks2D_str, bins, dilation_cycle,
                    optimizer_name, optimizer_str, criterion_name,
                    lr_modifier_name, lr_modifier_str, seed))


def parse_file_name(file_name):
    base = basename(file_name)
    num_blocks1D = base[base.find('1D')+2:base.find('_2D')].split('_')
    num_blocks2D = base[base.find('2D')+2:min(base.find('_dil'), base.find('_bins'))].split('_')

    num_blocks1D = [int(_) for _ in num_blocks1D]
    num_blocks2D = [int(_) for _ in num_blocks2D]

    return num_blocks1D, num_blocks2D
