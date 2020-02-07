import torch
import torch.sparse as sparse
from deeph3.util import pairwise_contact_probs, fill_diagonally_, mask_matrix_


seperation_ranges = {'short': [6, 11], 'medium': [12, 23],
                     'long': [24, float('inf')]}


def top_k_predictions(logits, k=2, contact_range='long', residue_ranges=None,
                      **kwargs):
    """

    Outputs the highest probability L/k predictions L is the length of the
    amino acid sequence.

    :param k:
    :param logits: The logits to generate probabilities from. Should have shape
                   (logits, n, n).
    :type logits: torch.Tensor
    :param kwargs:
    :return:
    """
    if residue_ranges is not None:
        if isinstance(residue_ranges[0], int):
            residue_ranges = [residue_ranges]
        seq_len = sum([ub - lb + 1 for lb, ub in residue_ranges])
    else:
        seq_len = logits.shape[1]

    probs = torch.Tensor(pairwise_contact_probs(logits, **kwargs))

    if contact_range != 'all':
        if contact_range not in seperation_ranges:
            msg = '{} is not a valid contact_range. The range must be in ' \
                  '{\'short\', \'medium\', \'long\', \'all\'}'
            raise ValueError(msg.format(contact_range))

        if residue_ranges is not None:
            for lb, ub in residue_ranges:
                i_mask = (probs[:, 0] >= lb).__and__(probs[:, 0] <= ub)
                j_mask = (probs[:, 1] >= lb).__and__(probs[:, 1] <= ub)
                mask = i_mask.__or__(j_mask)
                probs = probs[mask]

        # Filter down to residues within a given range
        lower_bound, upper_bound = seperation_ranges[contact_range]
        seperations = probs[:, 1] - probs[:, 0] - 1
        mask = (seperations >= lower_bound).__and__(seperations <= upper_bound)
        probs = probs[mask]

    if k is not None:
        top_k = probs[:seq_len // k]
        return top_k
    else:
        return probs


def top_k_contact_metrics(logits, dist_mat, k=5, contact_range='long',
                          residue_ranges=None, **kwargs):
    """Calculates metrics for the top L/k predicted contacts
    :param logits: The logits to generate probabilities from. Should have shape
                   (logits, n, n).
    :type logits: torch.Tensor
    :param true_dist_mat:
    :param k:
    :param kwargs: kwargs to pas to top_k_contacts
    :return:
    """
    dist_mat_clone = dist_mat.clone()  # Ensure original dist_mat is not modified

    # Get the predicted contacts in (i, j, prob) format
    top_k = top_k_predictions(logits, k=k, contact_range=contact_range,
                              residue_ranges=residue_ranges, **kwargs)
    predicted_contacts = top_k[top_k[:, 2] > 0.5]
    lower_bound, upper_bound = seperation_ranges[contact_range]

    # Ensure no diagonal value is non-zero
    dist_mat_clone[torch.eye(len(dist_mat_clone)).byte()] = 0

    # Remove residue pairs that are not in the contact range
    fill_diagonally_(dist_mat_clone, upper_bound + 1, fill_value=-1, fill_method='symmetric')
    fill_diagonally_(dist_mat_clone, lower_bound - 1, fill_value=-1, fill_method='between')

    if residue_ranges is not None:
        mask = torch.ones(dist_mat.shape[1])
        if isinstance(residue_ranges[0], int):
            mask[residue_ranges[0]:residue_ranges[1]+1] = 0
        else:
            for lb, ub in residue_ranges:
                mask[lb:ub+1] = 0
        mask_matrix_(dist_mat_clone, mask, not_mask_fill_value=-1)

    # Get the number of contacts on the entire matrix and divide by two because
    # of double counting caused by distance matrix symmetry
    total_contacts = len(dist_mat_clone[(dist_mat_clone < 8).__and__(dist_mat_clone > 0)])
    total_contacts //= 2

    # Turn sparse representation into dense, byte representation
    indices = torch.stack([predicted_contacts[:, 0], predicted_contacts[:, 1]]).long()
    values = torch.ones(predicted_contacts.shape[0])
    predicted_contact_mat = sparse.FloatTensor(indices, values, dist_mat.size())
    predicted_contact_mat = predicted_contact_mat.to_dense().byte()

    # Get the true contact matrix
    true_contact_mat = (dist_mat_clone < 8).__and__(dist_mat_clone > 0)

    # True positive, False positive, False negative calculations
    tp = len(predicted_contact_mat[predicted_contact_mat.__and__(true_contact_mat)])
    fp = int(predicted_contacts.shape[0]) - tp
    fn = total_contacts - tp

    # Calculate metrics
    if tp + fp == 0:
        precision = float('nan')
    else:
        precision = tp / float(tp + fp)
    if tp + fn == 0:
        recall = float('nan')
    else:
        recall = tp / float(tp + fn)
    if precision + recall == 0:
        f1 = float('nan')
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    # print(len(predicted_contact_mat[predicted_contact_mat]), tp, fp, fn, total_contacts)

    return torch.Tensor([precision, recall, f1])


def binned_dist_mat_contact_metrics(logits, true_binned_dist_mat, ang8_bin=8, **kwargs):
    """Gets the contact metrics of logits and a binned distance matrix label
    :param logits: The logits to generate probabilities from. Should have shape
                   (logits, n, n).
    :type logits: torch.Tensor
    :param ang8_bin:
        The index of the bin containing distances between 7.5 and 8 angstroms.
        It is assumed that every index prior is <8 angstroms
    :type ang8_bin: int
    """
    dist_mat = torch.Tensor(true_binned_dist_mat.shape).fill_(ang8_bin + 8)
    dist_mat[(true_binned_dist_mat <= ang8_bin).__and__(true_binned_dist_mat >= 0)] = 1
    return top_k_contact_metrics(logits, dist_mat, ang8_bin=ang8_bin, **kwargs)


def batch_binned_dist_mat_contact_metrics(logits, true_binned_dist_mats, ang8_bin=8, **kwargs):
    """Gets the average of each contact metric for a batch of binned distance matrices
    :param logits: The logits to generate probabilities from. Should have shape
                   (batch_size, logits, n, n).
    :type logits: torch.Tensor
    :param ang8_bin:
        The index of the bin containing distances between 7.5 and 8 angstroms.
        It is assumed that every index prior is <8 angstroms
    :type ang8_bin: int
    """
    metrics = torch.zeros(3)
    for i in range(logits.shape[0]):
        metrics += binned_dist_mat_contact_metrics(
            logits[i], true_binned_dist_mats[i], ang8_bin=ang8_bin, **kwargs)
    return metrics / logits.shape[0]


if __name__ == '__main__':
    import pickle
    from time import time
    from deeph3 import load_model
    from deeph3.util import get_logits_from_model, bin_distance_matrix, get_dist_bins
    checkpoint_file = './models/adam_opt_lr01_da/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_bins26_dil5_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patien.p'
    model = load_model(checkpoint_file)
    dist_mat = pickle.load(open('/Users/cguerra3/Rosetta_REU/deep-H3-loop-prediction/deeph3/data/dists/1sy6_trunc.p', 'rb'))
    dist_mat = bin_distance_matrix(dist_mat, bins=get_dist_bins(26))
    logits = get_logits_from_model(model, '/Users/cguerra3/Rosetta_REU/deep-H3-loop-prediction/deeph3/data/fastas/1sy6_trunc.fasta', chain_delimiter=True)
    # print(logits.unsqueeze(0))
    # print(logits.unsqueeze(0).shape)
    # print(batch_binned_dist_mat_contact_metrics(logits.unsqueeze(0), dist_mat.unsqueeze(0)))
    s_t = time()
    print(top_k_contact_metrics(logits, dist_mat, 1, contact_range='long', residue_ranges=[1, 6]))
    print(time() - s_t)

