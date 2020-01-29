import h5py
import pickle
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from deeph3.data_util import Batch
from deeph3.util import pad_data_to_same_shape, get_dist_bins, bin_distance_matrix, generate_pnet_dist_matrix


class H5ProteinNetDataset(data.Dataset):
    """
    Modified version of https://github.com/OpenProtein/openprotein
    """
    def __init__(self, filename, onehot_prim=True, num_dist_bins=26,
                 store_balanced_class_weights=True, max_seq_len=None):
        """
        :param filename: The h5 file for the ProteinNet data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        :param num_dist_bins:
            The number of bins to discretize the distance matrix into. If None,
            then the distance matrix remains continuous.
        """
        super(H5ProteinNetDataset, self).__init__()

        self.onehot_prim = onehot_prim
        self.filename = filename
        self.h5file = h5py.File(filename, 'r+')
        self.num_proteins, _ = self.h5file['primary'].shape
        self.bins = get_dist_bins(num_dist_bins) if num_dist_bins is not None else None
        self.store_balanced_class_weights = store_balanced_class_weights

        # Filter out sequences beyond the max length
        self.max_seq_len = max_seq_len
        self.valid_indices = None
        if max_seq_len is not None:
            self.valid_indices = self.get_valid_indices()
            self.num_proteins = len(self.valid_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        if self.valid_indices is not None:
            index = self.valid_indices[index]

        id_ = self.h5file['id'][index]
        seq_len = self.h5file['sequence_len'][index]

        # Get the attributes from a protein and cut off zero padding
        mask = torch.Tensor(self.h5file['mask'][index, :seq_len]).type(dtype=torch.uint8)
        prim = torch.Tensor(self.h5file['primary'][index, :seq_len]).type(dtype=torch.uint8)
        if self.onehot_prim:
            prim = F.one_hot(prim.long())
        evol = torch.Tensor(self.h5file['evolutionary'][index][:seq_len]).type(dtype=torch.float)
        tert = torch.Tensor(self.h5file['tertiary'][index][:seq_len]).type(dtype=torch.float)

        # Try to get the distance matrix from memory
        try:
            dist_mat = torch.Tensor(self.h5file['distance_mat'][index][:seq_len, :seq_len]).type(dtype=torch.float)
        except:
            dist_mat = generate_pnet_dist_matrix(tert, mask=mask)

        # Bin distance matrix with a mask
        if self.bins is not None:
            dist_mat = bin_distance_matrix(dist_mat, bins=self.bins, mask=mask)

        return id_, mask, prim, evol, tert, dist_mat

    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        for i, seq_len in enumerate(self.h5file['sequence_len']):
            if seq_len < self.max_seq_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return self.num_proteins

    def balanced_class_weights(self, show_progress=True, sample_percentage=1.0):
        """Estimates the weights for unbalanced classes in a onehot encoded dataset
        Uses the following equation to estimate weights:
        ``n_samples / (num_bins * torch.bincount(bin))``

        :param sample_percentage:
            The percentage of the data used when calculating the class weights
        """
        if self.bins is None:
            raise ValueError('Labels are continuous, class weights cannot be calculated.')
        if sample_percentage <= 0 or sample_percentage > 1:
            raise ValueError('Sample percentage must be between 0 and 1')

        num_bins = len(self.bins)
        try:
            for i, n_bins in enumerate(self.h5file['num_bins']):
                if n_bins == num_bins:
                    weights = self.h5file['class_weights'][i, :num_bins]
                    return weights
        except:
            pass

        count = torch.zeros([num_bins], dtype=torch.long)
        # Count the occurrences of each bin over the entire training dataset
        end_idx = int(len(self) * sample_percentage)
        if show_progress:
            print('Calculating class weights using {}% of the data...'.format(
                sample_percentage * 100))
        for i in tqdm(range((len(self)))[:end_idx], disable=(not show_progress)):
            _, mask, _, _, _, dist_mat = self[i]
            for bins in dist_mat:
                # Ignore bins that are -1
                bin_count = torch.bincount(bins.int() + 1)[1:]
                count[:bin_count.size(0)] += bin_count

        denominator = count * num_bins
        weights = sum(count) / denominator.float()

        # If a class was not in the dataset, weigh it as much as the highest
        # weighted class
        weights[denominator == 0] = -1
        weights[denominator == 0] = max(weights)
        pickle.dump(weights, open('weights_{}.p'.format(num_bins), 'wb'))

        if self.store_balanced_class_weights:
            max_bins = 50
            if 'num_bins' not in self.h5file.keys():
                self.h5file.create_dataset('num_bins', (1,), maxshape=(None,),
                                           dtype='float', fillvalue=0)
            if 'class_weights' not in self.h5file.keys():
                self.h5file.create_dataset('class_weights', (1, max_bins),
                                           maxshape=(None, max_bins),
                                           dtype='float', fillvalue=0)

            self.h5file['num_bins'].resize((len(self.h5file['num_bins']) + 1, ))
            self.h5file['class_weights'].resize((len(self.h5file['class_weights']) + 1, max_bins))
            index = len(self.h5file['num_bins']) - 1
            self.h5file['num_bins'][index] = num_bins
            self.h5file['class_weights'][index, :num_bins] = weights
            self.h5file.close()
            self.h5file = h5py.File(self.filename, 'r+')
        return weights

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return H5ProteinNetBatch(zip(*samples))


class H5ProteinNetBatch(Batch):
    def __init__(self, data):
        (self.id_, self.mask, self.primary, self.evolutionary, self.tertiary, self.distance_mat) = data

    def features(self):
        """Gets the PSSM data of the batch with zero padding"""
        X = self.evolutionary
        X = pad_data_to_same_shape(X, pad_value=0)

        # Switch shape from [batch, timestep/length, filter/channel]
        #                to [batch, filter/channel, timestep/length]
        return X.transpose(1, 2).contiguous()

    def labels(self):
        """Gets the distance matrix data of the batch with -1 padding"""
        y = self.distance_mat
        y = pad_data_to_same_shape(y, pad_value=-1)
        return y.long()

    def batch_mask(self):
        """Gets the mask data of the batch with zero padding"""
        masks = self.mask
        masks = pad_data_to_same_shape(masks, pad_value=0)
        return masks


def h5_proteinnet_dataloader(filename, batch_size=8, max_seq_len=None, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError('Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(dict(collate_fn=H5ProteinNetDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(H5ProteinNetDataset(filename, max_seq_len=max_seq_len), **kwargs)


if __name__ == '__main__':
    def main():
        from tqdm import tqdm
        from deeph3.viz import heatmap2d
        dataloader = h5_proteinnet_dataloader(
            '/Users/cguerra3/Rosetta_REU/deep-H3-loop-prediction/deeph3/data/casp11_h5/testing.h5', 8)

        for _ in tqdm(dataloader):
            print(_.labels().shape)
            print(_.features().shape)
            heatmap2d(_.labels()[-1], title=_.id_[-1])
    main()

