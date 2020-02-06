import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm import tqdm
from os.path import basename, join
from deeph3 import load_checkpoint_dict, save_train_checkpoint
from deeph3.evaluator import test_model
from deeph3.util import time_diff, get_probs_from_model, binned_matrix
from deeph3.data_util import Batch, to_device, get_default_device
from deeph3.contact_metrics import batch_binned_dist_mat_contact_metrics
from pathlib import Path


def train_model(train_loader, model, optimizer, lr_modifier, criterion, writer,
                start_batch=0, save_func=None, save_every=None, max_iter=None,
                device=None, metrics=None, metric_labels=None, epoch=0):
    """Trains a model on a training set

    :param train_loader:
    :type train_loader: torch.utils.data.DataLoader
    :param model:
    :type model: torch.Model
    :param optimizer:
    :type optimizer:
    :param criterion:
    :type criterion:
    :param metrics:
        A function that takes in a batch of model output and the target values
        and then returns a tensor of metric values. For example, this function
        may return a tensor with three elements:
        torch.Tensor([average_precision, average_accuracy, average_f1])
    :param metric_labels:
        The label of each metric returned by metrics. As per the previous
        example, a suitable list of labels would be:
        ['average_precision', 'average_accuracy', 'average_f1']
    """
    print('\nTraining Model...')
    if save_every is not None and save_func is None:
        raise ValueError('save_func cannot be none if save_every is not None')
    if save_every is None:
        save_every = len(train_loader.dataset)
    if device is None:
        device = get_default_device()

    print('Using {} as device'.format(str(device).upper()))
    model = to_device(model, device)

    model.train()
    output_names = ["dist", "omega", "theta", "phi"]
    running_loss = torch.zeros(5)
    running_metrics = 0.0
    total_batches = max_iter
    if max_iter is None:
        total_batches = (len(train_loader.dataset) //
                         train_loader.batch_size) + 1
        max_iter = float('Inf')
    for i, data in tqdm(enumerate(train_loader), total=total_batches):
        if i >= max_iter:
            break
        # Skip batch if it has already been seen
        if i < start_batch:
            continue

        batch_start = time()
        if issubclass(type(data), Batch):
            inputs, labels = data.data()
        else:
            inputs, labels = data

        inputs = to_device(inputs, device)
        labels = [to_device(label, device) for label in labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            outputs = model(inputs).transpose(0, 1)
            losses = [crit(output, label) for crit, output,
                      label in zip(criterion, outputs, labels)]
            loss = sum(losses)
            losses.append(loss)

            loss.backward()
            optimizer.step()

            return outputs, [float(loss.item()) for loss in losses]

        outputs, batch_loss = handle_batch()
        for loss_i, loss in enumerate(batch_loss):
            running_loss[loss_i] += loss

        if metrics is not None:
            batch_metrics = metrics(outputs[0], labels[0])
            running_metrics += batch_metrics

            print('\nTotal time for batch of {} samples: {}'.format(
                len(labels[0]), time_diff(batch_start, time())))
            print('batch {:5d}: loss: {:.3f}, metrics: {}'.format(
                (i + 1), batch_loss[-1], batch_metrics))
        else:
            print('\nTotal time for batch of {} samples: {}'.format(
                len(labels[0]), time_diff(batch_start, time())))
            print('batch {:5d}: loss: {:.3f}'.format((i + 1), batch_loss))
        # print('\nTotal time for batch of {} samples: {}'.format(len(labels), time_diff(batch_start, time())))
        # print('batch {:5d}: loss: {:.3f}'.format((i + 1), batch_loss))

        if i != 0 and save_every != 0 and i % save_every == 0:
            avg_loss = running_loss / (i + 1)
            avg_metrics = running_metrics / (i + 1)
            if metric_labels is not None:
                avg_metrics = {l: v.item()
                               for l, v in zip(metric_labels, avg_metrics)}

            save_func(model=model, optimizer=optimizer, criterion=criterion,
                      start_batch=i, train_loader=train_loader,
                      avg_train_loss=avg_loss, lr_modifier=lr_modifier,
                      training_metrics=avg_metrics)

    lr_modifier.step(running_loss[-1])

    avg_loss = running_loss / (i + 1)
    writer.add_scalars("train_loss", dict(
        zip(output_names + ["total"], avg_loss)), global_step=epoch)

    avg_metrics = running_metrics / (i + 1)
    if metric_labels is not None:
        avg_metrics = {l: v.item() for l, v in zip(metric_labels, avg_metrics)}

    if metrics is not None:
        print('\nAverage loss: {}, Average metrics: {}'.format(
            avg_loss, avg_metrics))
    else:
        print('\nAverage loss: {}'.format(avg_loss))

    save_func(model=model, optimizer=optimizer, criterion=criterion,
              start_batch=i, train_loader=train_loader,
              avg_train_loss=avg_loss, lr_modifier=lr_modifier,
              training_metrics=avg_metrics)

    return avg_loss


def h5_trainer(train_file_path, validation_file_path, in_planes=20,
               num_blocks1D=3, num_blocks2D=10, dilation_cycle=0, batch_size=4, epochs=20,
               save_every=100, out_fasta=None, max_validation_iter=None,
               max_training_iter=None, optimizer_type='SGD',
               optimizer_kwargs=None, dataset_type='ab', models_dir=None,
               criterion_type='CrossEntropyLoss',
               lr_modifer_type='ReduceLROnPlateau', lr_modifier_kwargs=None,
               k=2, **kwargs):
    checkpoint = load_checkpoint_dict(
        in_planes=in_planes,
        num_blocks1D=num_blocks1D,
        num_blocks2D=num_blocks2D,
        dilation_cycle=dilation_cycle,
        train_h5_file_path=train_file_path,
        validation_h5_file_path=validation_file_path,
        lr_modifier_type=lr_modifer_type,
        lr_modifier_kwargs=lr_modifier_kwargs,
        batch_size=batch_size,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        dataset_type=dataset_type,
        models_dir=models_dir,
        criterion_type=criterion_type,
        **kwargs)

    model = checkpoint['model']
    train_loader = checkpoint['train_loader']
    validation_loader = checkpoint['validation_loader']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    training_metrics = checkpoint['training_metrics']
    validation_metrics = checkpoint['validation_metrics']
    dilation_cycle = checkpoint['dilation_cycle']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    lr_modifier = checkpoint['lr_modifier']
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch']
    batch_size = checkpoint['batch_size']
    max_validation_seq_len = checkpoint['max_validation_seq_len']
    max_training_seq_len = checkpoint['max_training_seq_len']

    seed = checkpoint['seed']
    torch.manual_seed(seed)

    print(model)

    metric_labels = ['precision_L/{}', 'recall_L/{}', 'f1_L/{}']
    metric_labels = [_.format(k) for _ in metric_labels]

    writer = SummaryWriter(join(models_dir, "tensorboard"))

    for epoch in range(start_epoch, epochs):
        def metrics(logits, label): return batch_binned_dist_mat_contact_metrics(
            logits, label, k=k)
        save_func = create_batch_saver(
            writer=writer,
            epoch=epoch,
            dilation_cycle=dilation_cycle,
            validation_loader=validation_loader,
            all_training_losses=training_losses,
            all_validation_losses=validation_losses,
            seed=seed,
            batch_size=batch_size,
            max_training_seq_len=max_training_seq_len,
            max_validation_seq_len=max_validation_seq_len,
            optimizer_kwargs=optimizer_kwargs,
            lr_modifier_kwargs=lr_modifier_kwargs,
            dataset_type=dataset_type,
            out_fasta=out_fasta,
            max_iter=max_validation_iter,
            in_planes=in_planes,
            models_dir=models_dir,
            all_training_metrics=training_metrics,
            all_validation_metrics=validation_metrics,
            metrics=metrics,
            metric_labels=metric_labels)

        train_model(train_loader=train_loader, model=model, optimizer=optimizer,
                    criterion=criterion, writer=writer, lr_modifier=lr_modifier,
                    start_batch=start_batch,
                    save_every=save_every, save_func=save_func,
                    max_iter=max_training_iter,
                    metrics=metrics,
                    metric_labels=metric_labels,
                    epoch=epoch)

    writer.close()


def create_batch_saver(writer, epoch, dilation_cycle, validation_loader, all_training_losses,
                       all_training_metrics, all_validation_metrics,
                       all_validation_losses, seed, batch_size,
                       max_training_seq_len, max_validation_seq_len,
                       optimizer_kwargs, lr_modifier_kwargs, in_planes=20,
                       out_fasta=None, max_iter=None, metrics=None,
                       metric_labels=None, models_dir=None, dataset_type='pnet'):
    """Generates a function to save a ProteinNet model mid-training

    This function is needed to initialize constants that save_func does not
    take as input but needs for saving a model.
    :param out_fasta: The fasta file to use when generating an output distance matrix image.
                      Images are not output if None.
    """
    def save_func(model, optimizer, lr_modifier, criterion, start_batch,
                  train_loader, avg_train_loss, training_metrics):
        """Performs validation and saves a ProteinNet model mid-training"""
        avg_val_loss, validation_metrics = test_model(
            validation_loader, model, criterion, writer, max_iter=max_iter,
            metrics=metrics, metric_labels=metric_labels, epoch=epoch)

        # Update loss
        all_training_losses.append([epoch, start_batch, avg_train_loss])
        all_validation_losses.append([epoch, start_batch, avg_val_loss])
        all_training_metrics.append(training_metrics)
        all_validation_metrics.append(validation_metrics)
        print('Saving model...')
        save_train_checkpoint(
            train_loader=train_loader,
            validation_loader=validation_loader,
            model=model,
            dilation_cycle=dilation_cycle,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_modifier=lr_modifier,
            lr_modifier_kwargs=lr_modifier_kwargs,
            epoch=epoch,
            batch=start_batch+1,
            training_losses=all_training_losses,
            validation_losses=all_validation_losses,
            seed=seed,
            batch_size=batch_size,
            max_training_seq_len=max_training_seq_len,
            max_validation_seq_len=max_validation_seq_len,
            dataset_type=dataset_type,
            models_dir=models_dir,
            training_metrics=all_training_metrics,
            validation_metrics=all_validation_metrics)

        if out_fasta is not None:
            val_file = basename(
                validation_loader.dataset.filename).split('.')[0]
            train_file = basename(train_loader.dataset.filename).split('.')[0]
            base_name = basename(out_fasta).split('.')[0]

            file_name = 'predicted_{}_epoch{}_batches{}_batchsize{}_{}_{}_{}.png'.format(
                base_name, epoch, start_batch+1, batch_size, dataset_type, train_file, val_file)
            title = 'Predicted {}'.format(base_name)
            model.eval()
            if in_planes == 21:
                probs = get_probs_from_model(
                    model, out_fasta, chain_delimiter=True)
            else:
                probs = get_probs_from_model(model, out_fasta)
            mat = binned_matrix(probs, are_logits=False)

            msg = 'Writing predicted distance matrix to {}...'
            print(msg.format(Path(file_name).absolute()))

    return save_func
