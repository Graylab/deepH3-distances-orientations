import torch
import pandas as pd
from time import time
from tqdm import tqdm
from deeph3.viz import heatmap2d
from deeph3.util import time_diff, binned_matrix, binned_mat_to_values
from deeph3.data_util import Batch, H5AntibodyDataset, to_device, get_default_device
from deeph3.contact_metrics import batch_binned_dist_mat_contact_metrics
from deeph3.data_util.H5AntibodyDataset import h5_antibody_dataloader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from os.path import join


def test_model(test_loader, model, criterion, writer, max_iter=None, device=None,
               metrics=None, metric_labels=None, epoch=0):
    """Tests the model on a test set
    """
    print('\nTesting Model...')
    if device is None:
        device = get_default_device()

    print('Using {} as device'.format(str(device).upper()))
    model = to_device(model, device)

    running_metrics = 0.0
    metric_counts = 0  # Counts the number of valid results for each metric
    # torch.no_grad used to suspend to suspend autograd, saving time + memory
    with torch.no_grad():
        model.eval()  # Set model to eval mode

        output_names = ["dist", "omega", "theta", "phi"]
        running_loss = torch.zeros(5)
        total_batches = max_iter
        if max_iter is None:
            total_batches = (len(test_loader.dataset) //
                             test_loader.batch_size) + 1
            max_iter = float('Inf')
        for i, data in tqdm(enumerate(test_loader), total=total_batches):
            if i >= max_iter:
                break

            batch_start = time()
            if issubclass(type(data), Batch):
                inputs, labels = data.data()
                ids = data.id()
            else:
                inputs, labels = data
                ids = data.id_

            labels = [to_device(label, device) for label in labels]

            # forward
            inputs = to_device(inputs, device)
            outputs = model(inputs).transpose(0, 1)


            for batch_i in range(outputs.shape[1]):
                binned_mat = binned_matrix(outputs.transpose(0, 1)[batch_i])
                for output_name, output in zip(output_names, binned_mat):
                    fig = plt.figure()
                    heatmap2d(output, title=output_name, color_min=0,
                              color_max=25, keep_open=True)
                    plt.tight_layout()
                    plt.close()
                    writer.add_figure(ids[batch_i].decode(
                        "utf-8") + "/" + output_name, fig, global_step=epoch)

            losses = [crit(output, label).item() for crit, output,
                      label in zip(criterion, outputs, labels)]
            losses.append(sum(losses))
            for loss_i, loss in enumerate(losses):
                running_loss[loss_i] += float(loss)

            if metrics is not None:
                batch_metrics = metrics(outputs[0], labels[0])
                metric_counts += 1 - torch.isnan(batch_metrics)
                batch_metrics[torch.isnan(batch_metrics)] = 0
                running_metrics += batch_metrics

                print('\nTotal time for batch of {} samples: {}'.format(
                    len(labels[0]), time_diff(batch_start, time())))
                print('batch {:5d}: loss: {:.3f}, metrics: {}'.format(
                    (i + 1), losses[-1], batch_metrics))
            else:
                print('\nTotal time for batch of {} samples: {}'.format(
                    len(labels[0]), time_diff(batch_start, time())))
                print('batch {:5d}: loss: {:.3f}'.format(
                    (i + 1), losses[-1]))

        avg_loss = running_loss / (i + 1)
        writer.add_scalars("validation_loss", dict(
            zip(output_names + ["total"], avg_loss)), global_step=epoch)

        avg_metrics = running_metrics / metric_counts.float()
        if metric_labels is not None:
            avg_metrics = {l: v.item()
                           for l, v in zip(metric_labels, avg_metrics)}
            writer.add_scalars("average_metrics",
                               avg_metrics, global_step=epoch)
        print('\nAverage loss: {}'.format(avg_loss))

    return avg_loss, avg_metrics


def get_model_metric_performances(test_loader, model, metrics,
                                  metric_labels=None, device=None):
    print('\nCalculating model metrics...')
    if device is None:
        device = get_default_device()

    print('Using {} as device'.format(str(device).upper()))
    model = to_device(model, device)

    running_metrics = 0.0
    metric_counts = 0  # Counts the number of valid results for each metric
    all_metrics = []
    # torch.no_grad used to suspend to suspend autograd, saving time + memory
    with torch.no_grad():
        model.eval()  # Set model to eval mode
        total_batches = (len(test_loader.dataset) //
                         test_loader.batch_size) + 1
        for i, data in tqdm(enumerate(test_loader), total=total_batches):
            batch_start = time()
            print(data.id_)
            if issubclass(type(data), Batch):
                inputs, labels = data.data()
            else:
                inputs, labels = data

            inputs = to_device(inputs, device)
            labels = [to_device(label, device) for label in labels]

            outputs = model(inputs).transpose(0, 1)
            batch_metrics = metrics(data, outputs[0], labels[0])
            all_metrics.append(list(batch_metrics))

            metric_counts += 1 - torch.isnan(batch_metrics)
            batch_metrics[torch.isnan(batch_metrics)] = 0
            running_metrics += batch_metrics

            print('\nTotal time for batch of {} samples: {}'.format(
                len(labels[0]), time_diff(batch_start, time())))
            print('batch {:5d}: metrics: {}'.format((i + 1), batch_metrics))

            break

        avg_metrics = running_metrics / metric_counts.float()
        if metric_labels is not None:
            avg_metrics = {l: v.item()
                           for l, v in zip(metric_labels, avg_metrics)}
        print('\nAverage metrics: {}'.format(avg_metrics))

    return avg_metrics, all_metrics


def get_all_ab_contact_metrics(antibody_h5file, model, h3_only=False):
    test_loader = h5_antibody_dataloader(
        antibody_h5file, batch_size=1, shuffle=False)

    ks = [10, 5, 2, 1, None]
    contact_ranges = ['medium', 'short', 'long']
    columns = ['k', 'contact_range', 'precision_L/k', 'recall_L/k', 'f1_L/k']
    all_avg_metrics_df = pd.DataFrame(columns=columns)
    all_metrics_columns = ['id', 'h3_len'] + columns
    all_metrics_df = pd.DataFrame(columns=all_metrics_columns)
    for contact_range in contact_ranges:
        for k in ks:
            columns = [_.format(k) for _ in columns]
            print(contact_range, k)
            if h3_only:
                def metrics(batch, logits, label): return batch_binned_dist_mat_contact_metrics(
                    logits, label, k=k, contact_range=contact_range, residue_ranges=batch.h3)
            else:
                def metrics(batch, logits, label): return batch_binned_dist_mat_contact_metrics(
                    logits, label, k=k, contact_range=contact_range)

            avg_metrics, all_metrics = get_model_metric_performances(
                test_loader, model, metrics=metrics)
            avg_metrics = [float(_) for _ in avg_metrics]

            # Get antibody information: its id, and h3 loop length
            ab_info = [[ab.id_[0].decode('utf-8'), ab.h3[0][1] - ab.h3[0][0] + 1]
                       for ab in test_loader]
            # Add all the information and metrics to the main dataframe
            all_metrics_list = [info + [k, contact_range] + [float(_) for _ in m]
                                for info, m in zip(ab_info, all_metrics)]
            all_metrics_df = all_metrics_df.append(pd.DataFrame(all_metrics_list,
                                                                columns=all_metrics_columns))
            print(all_metrics_df)

            # Update average metrics for the specific k, contact_range combo
            k_metrics = pd.DataFrame([[k, contact_range] + avg_metrics],
                                     columns=columns)
            all_avg_metrics_df = all_avg_metrics_df.append(k_metrics)
        print(all_avg_metrics_df)
    return all_avg_metrics_df, all_metrics_df


def get_h3_contact_metrics(df):
    """
    :param df:
    :return:
    """
    df.k = df.k.fillna(0)  # Replace NaNs (meaning no k value) with zeros

    metric_names = ['precision_L/k', 'recall_L/k', 'f1_L/k']
    columns = ['h3_len', 'k', 'contact_range'] + metric_names
    h3_metrics_df = pd.DataFrame(columns=columns)
    for h3_len in df.h3_len.unique():
        for k in df.k.unique():
            for contact_range in df.contact_range.unique():
                mask = (df.h3_len == h3_len).__and__(df.k == k)
                mask = mask.__and__(df.contact_range == contact_range)

                metric_avgs = [df[mask][m].mean() for m in metric_names]
                row_df = pd.DataFrame([[h3_len, k, contact_range] + metric_avgs],
                                      columns=columns)
                h3_metrics_df = h3_metrics_df.append(row_df)
    return h3_metrics_df


if __name__ == '__main__':
    def main():
        import pickle
        import matplotlib.pyplot as plt
        from deeph3.viz import heatmap2d
        from deeph3 import load_model

        # csv = '/Users/cguerra3/Rosetta_REU/deep-H3-loop-prediction/deeph3/all_full_antibody_contact_metrics_1.csv'
        # all_metrics = pd.read_csv(csv)
        '''
        df = get_h3_contact_metrics(all_metrics)

        for h3_len in sorted(df.h3_len.unique()):
            for contact_range in ['medium']:  #df.contact_range.unique():
                mask = (df.h3_len == h3_len).__and__(df.contact_range == contact_range)
                print(df[mask])
                df[mask].plot.line(x='k', y='f1_L/k')
                plt.title(str(contact_range) + ' ' + str(h3_len))
        plt.show()
        '''
        #get_h3_metrics_from_dataframe(None, '/Users/cguerra3/Rosetta_REU/deep-H3-loop-prediction/deeph3/data/antibody_h5/antibody_test.h5')
        checkpoint_file = 'deeph3/models/adam_opt_lr01_dil2/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_dil5_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'
        antibody_h5file = 'deeph3/data/antibody_dataset/h5s3/antibody_validation.h5'

        from deeph3.contact_metrics import top_k_predictions
        from deeph3.util import fill_diagonally_, mask_matrix_

        model = load_model(checkpoint_file)
        # test_loader = h5_antibody_dataloader(antibody_h5file, batch_size=4, shuffle=False)

        '''
        contact_range = 'long'
        k = None
        seperation_ranges = {'short': [6, 11], 'medium': [12, 23],
                             'long': [24, float('inf')]}
        num_in_contact = 0
        total = 0
        for data in test_loader:
            features, labels = data.data()
            #logits = model(features)

            residue_ranges = data.h3
            #if residue_ranges[0][1] - residue_ranges[0][0] + 1 > 17:
            #    continue
            total += 1
            dist_mat = labels[0]

            dist_mat_clone = dist_mat.clone()  # Ensure original dist_mat is not modified
            dist_mat_clone = (dist_mat_clone <= 8).int()

            # Get the predicted contacts in (i, j, prob) format
            #predicted_contacts = top_k_contacts(logits, k=k, contact_range=contact_range,
            #                                    residue_ranges=residue_ranges)
            lower_bound, upper_bound = seperation_ranges[contact_range]

            # Ensure no diagonal value is non-zero
            dist_mat_clone[torch.eye(len(dist_mat_clone)).byte()] = 0
            # Include only residues in the residue range
            if residue_ranges is not None:
                mask = torch.ones(dist_mat.shape[1])
                if isinstance(residue_ranges[0], int):
                    mask[residue_ranges[0]:residue_ranges[1]+1] = 0
                else:
                    for lb, ub in residue_ranges:
                        mask[lb:ub+1] = 0
                mask_matrix_(dist_mat_clone, mask, not_mask_fill_value=-1)
            # Remove residue pairs that are not in the contact range
            fill_diagonally_(dist_mat_clone, upper_bound + 1, fill_value=-1, fill_method='symmetric')
            fill_diagonally_(dist_mat_clone, lower_bound - 1, fill_value=-1, fill_method='between')

            for _ in dist_mat_clone:
                if 1 in _:
                    num_in_contact += 1
                    break
            #heatmap2d(dist_mat_clone)

        print(total)
        print(num_in_contact)
        '''
        contact_metrics, all_contact_metrics = get_all_ab_contact_metrics(
            antibody_h5file, model, h3_only=True)

        # pickle.dump(contact_metrics, open('contact_metrics_full_ab.p', 'wb'))
        contact_metrics.to_csv('../full_antibody_contact_metrics_h3.csv')
        # pickle.dump(all_contact_metrics, open('all_contact_metrics_full_ab.p', 'wb'))
        all_contact_metrics.to_csv(
            '../all_full_antibody_contact_metrics_h3.csv')

        contact_metrics, all_contact_metrics = get_all_ab_contact_metrics(
            antibody_h5file, model, h3_only=False)

        # pickle.dump(contact_metrics, open('contact_metrics_full_ab.p', 'wb'))
        contact_metrics.to_csv('../full_antibody_contact_metrics.csv')
        # pickle.dump(all_contact_metrics, open('all_contact_metrics_full_ab.p', 'wb'))
        all_contact_metrics.to_csv('../all_full_antibody_contact_metrics.csv')
    main()
