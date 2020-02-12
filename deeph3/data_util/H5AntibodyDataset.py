import h5py
import pickle
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from deeph3.util import pad_data_to_same_shape, get_dist_bins, get_omega_bins, get_theta_bins, get_phi_bins, bin_distance_matrix, bin_euler_matrix, bin_dist_angle_matrix


class H5AntibodyDataset(data.Dataset):
    def __init__(self, filename, onehot_prim=True, num_bins=26,
                 store_balanced_class_weights=True, max_seq_len=None):
        """
        :param filename: The h5 file for the antibody data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        :param num_bins:
            The number of bins to discretize the distance matrix into. If None,
            then the distance matrix remains continuous.
        """
        super(H5AntibodyDataset, self).__init__()

        self.onehot_prim = onehot_prim
        self.filename = filename
        self.h5file = h5py.File(filename, 'r+')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape
        self.dist_bins = get_dist_bins(
            num_bins) if num_bins is not None else None
        self.omega_bins = get_omega_bins(
            num_bins) if num_bins is not None else None
        self.theta_bins = get_theta_bins(
            num_bins) if num_bins is not None else None
        self.phi_bins = get_phi_bins(
            num_bins) if num_bins is not None else None
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
        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len

        # Get the attributes from a protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]

        # Convert to torch tensors
        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)

        # Get CDR loops
        h3 = self.h5file['h3_range'][index]

        if self.onehot_prim:
            heavy_prim = F.one_hot(heavy_prim.long())
            light_prim = F.one_hot(light_prim.long())

        # Try to get the distance matrix from memory
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][index][:4, :total_seq_len, :total_seq_len]
            dist_angle_mat = torch.Tensor(
                dist_angle_mat).type(dtype=torch.float)
        except Exception:
            #euler_mat = generate_pnet_euler_matrix(tert, mask=mask)
            raise ValueError('Output matrix not defined')

        # Bin distance matrix with a mask
        dist_angle_mat = bin_dist_angle_matrix(dist_angle_mat)

        return id_, heavy_prim, light_prim, dist_angle_mat, h3

    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        for i in range(self.h5file['heavy_chain_seq_len'].shape[0]):
            h_len = self.h5file['heavy_chain_seq_len'][i]
            l_len = self.h5file['light_chain_seq_len'][i]
            total_seq_len = h_len + l_len
            if total_seq_len < self.max_seq_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return self.num_proteins

    def balanced_class_weights(self, show_progress=True, sample_percentage=1.0, use_last=True):
        """Estimates the weights for unbalanced classes in a onehot encoded dataset
        Uses the following equation to estimate weights:
        ``n_samples / (num_bins * torch.bincount(bin))``

        :param sample_percentage:
            The percentage of the data used when calculating the class weights
        """
        if use_last and 'num_bins' in self.h5file.keys() and 'class_weights' in self.h5file.keys():
            print("Using latest class weights from {}".format(self.h5file))
            index = len(self.h5file['num_bins']) - 1
            num_bins = int(self.h5file['num_bins'][index])
            weights = self.h5file['class_weights'][index, :, :num_bins]

            return weights

        if self.dist_bins is None:
            raise ValueError(
                'Distance labels are continuous, class weights cannot be calculated.')
        if self.omega_bins is None:
            raise ValueError(
                'Omega labels are continuous, class weights cannot be calculated.')
        if self.theta_bins is None:
            raise ValueError(
                'Theta labels are continuous, class weights cannot be calculated.')
        if self.phi_bins is None:
            raise ValueError(
                'Phi labels are continuous, class weights cannot be calculated.')
        if sample_percentage <= 0 or sample_percentage > 1:
            raise ValueError('Sample percentage must be between 0 and 1')

        num_bins = len(self.dist_bins)
        try:
            for i, n_bins in enumerate(self.h5file['num_bins']):
                if n_bins == num_bins:
                    weights = self.h5file['class_weights'][i, 4, :num_bins]
                    return weights
        except Exception:
            pass

        count = torch.zeros([4, num_bins], dtype=torch.long)
        # Count the occurrences of each bin over the entire training dataset
        end_idx = int(len(self) * sample_percentage)
        if show_progress:
            print('Calculating class weights using {}% of the data...'.format(
                sample_percentage * 100))
        for i in tqdm(range(len(self))[:end_idx], disable=(not show_progress)):
            _, _, _, dist_angle_mat, _ = self[i]
            for j, mat in enumerate(dist_angle_mat):
                for bins in mat:
                    # Ignore bins that are -1
                    bin_count = torch.bincount(bins.int() + 1)[1:]
                    count[j, :bin_count.size(0)] += bin_count

        denominator = count * num_bins
        weights = torch.div(torch.sum(count, dim=1).unsqueeze(
            1).float(), denominator.float())

        # If a class was not in the dataset, weigh it as much as the highest
        # weighted class
        weights[denominator == 0] = -1
        weights[denominator == 0] = torch.max(weights, dim=1, keepdim=True)[
            0].expand(weights.shape)[denominator == 0]
        pickle.dump(weights, open('weights_{}.p'.format(num_bins), 'wb'))

        if self.store_balanced_class_weights:
            max_bins = 50
            if 'num_bins' not in self.h5file.keys():
                self.h5file.create_dataset('num_bins', (1,), maxshape=(None,),
                                           dtype='float', fillvalue=0)
            if 'class_weights' not in self.h5file.keys():
                self.h5file.create_dataset('class_weights', (1, 4, max_bins),
                                           maxshape=(None, 4, max_bins),
                                           dtype='float', fillvalue=0)

            self.h5file['num_bins'].resize(
                (len(self.h5file['num_bins']) + 1, ))
            self.h5file['class_weights'].resize(
                (len(self.h5file['class_weights']) + 1, 4, max_bins))
            index = len(self.h5file['num_bins']) - 1
            self.h5file['num_bins'][index] = num_bins
            self.h5file['class_weights'][index, :, :num_bins] = weights
            self.h5file.close()
            self.h5file = h5py.File(self.filename, 'r+')
        return weights

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return H5AntibodyBatch(zip(*samples)).data()


class H5AntibodyBatch:
    def __init__(self, batch_data):
        (self.id_, self.heavy_prim, self.light_prim,
         self.dist_angle_mat, self.h3) = batch_data

    def data(self):
        return self.features(), self.labels()

    def features(self):
        """Gets the one-hot encoding of the sequences with a feature that
        delimits the chains"""
        X = [torch.cat(_, 0) for _ in zip(self.heavy_prim, self.light_prim)]
        X = pad_data_to_same_shape(X, pad_value=0)

        # Add chain delimiter
        X = F.pad(X, (0, 1, 0, 0, 0, 0))
        for i, h_prim in enumerate(self.heavy_prim):
            X[i, len(h_prim)-1, X.shape[2]-1] = 1

        # Switch shape from [batch, timestep/length, filter/channel]
        #                to [batch, filter/channel, timestep/length]
        return X.transpose(1, 2).contiguous()

    def labels(self):
        """Gets the distance matrix data of the batch with -1 padding"""
        label_mat = pad_data_to_same_shape(
            self.dist_angle_mat, pad_value=-1).transpose(0, 1).long()

        return label_mat

    def batch_mask(self):
        """Gets the mask data of the batch with zero padding"""
        '''Code to use when masks are added
        masks = self.mask
        masks = pad_data_to_same_shape(masks, pad_value=0)
        return masks
        '''
        raise NotImplementedError(
            'Masks have not been added to antibodies yet')


def h5_antibody_dataloader(filename, batch_size=1, max_seq_len=None, num_bins=26, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(
        dict(collate_fn=H5AntibodyDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(H5AntibodyDataset(filename, num_bins=num_bins, max_seq_len=max_seq_len), **kwargs)

