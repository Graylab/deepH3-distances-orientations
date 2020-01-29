import torch
import h5py
import numpy as np
from os.path import isfile
from deeph3.preprocess.pnet_text_parser import read_record
from deeph3.util import generate_pnet_dist_matrix
from tqdm import tqdm


def pnet_seq_info(file_path):
    """Gets information on the sequences in a ProteinNet file
    :return: The number of sequences, the min and max sequence length in a three
             tuple (num_seqs, min_len, max_len)
    """
    num_seqs = 0
    max_len = -1
    min_len = float('inf')
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            if line == '[PRIMARY]\n':
                num_seqs += 1
                seq = f.readline().replace('\n', '')
                seq_len = len(seq)
                max_len = max(max_len, seq_len)
                min_len = min(min_len, seq_len)
            line = f.readline()
    return num_seqs, min_len, max_len


def pnet_to_h3(file_path, out_file_path, num_evo_entries=20, overwrite=False,
               print_progress=False):
    """Converts a ProteinNet text file into a h5 file without the secondary
    structure data"""
    if isfile(out_file_path) and not overwrite:
        print('{} already exists. Set overwrite to true to overwrite it.'.format(out_file_path))
        return
    num_seqs, _, max_seq_len = pnet_seq_info(file_path)
    out = h5py.File(out_file_path)

    # Create relevant datasets
    id_set = out.create_dataset('id', (num_seqs,), dtype='S25')
    seq_len_set = out.create_dataset('sequence_len', (num_seqs,), fillvalue=0,
                                     maxshape=(None,), dtype='uint16')
    prim_set = out.create_dataset('primary', (num_seqs, max_seq_len), fillvalue=0,
                                  maxshape=(None, max_seq_len), dtype='uint8')
    tert_set = out.create_dataset('tertiary', (num_seqs, max_seq_len, 9), fillvalue=0.,
                                  maxshape=(None, max_seq_len, 9), dtype='float')
    mask_set = out.create_dataset('mask', (num_seqs, max_seq_len), fillvalue=0,
                                  maxshape=(None, max_seq_len), dtype='uint8')
    evol_set = out.create_dataset('evolutionary', (num_seqs, max_seq_len, num_evo_entries),
                                  fillvalue=0., maxshape=(None, max_seq_len, num_evo_entries),
                                  dtype='float')
    dist_set = out.create_dataset('distance_mat', (num_seqs, max_seq_len, max_seq_len),
                                  maxshape=(None, max_seq_len, max_seq_len),
                                  dtype='float', fillvalue=-1)

    # Add headers to datasets that store a list of features for each amino acid
    tert_set.attrs['header'] = ['N_x', 'N_y', 'N_z',
                                'CA_x', 'CA_y', 'CA_z',
                                'C_x', 'C_y', 'C_z']
    evol_set.attrs['header'] = [_ for _ in 'ACDEFGHIKLMNPQRSTVWY']

    input_file = open(file_path, 'r')
    for cur_record in tqdm(range(num_seqs), disable=(not print_progress)):
        record = read_record(input_file, num_evo_entries)
        if record is not None:
            seq_len = len(record['primary'])

            prim_padded = np.zeros(max_seq_len)
            tert_padded = np.zeros((max_seq_len, 9))
            mask_padded = np.zeros(max_seq_len, dtype=bool)
            evol_padded = np.zeros((max_seq_len, num_evo_entries))

            # Reshape data such that the entry is of shape (seq_len, features)
            tert = np.ravel(np.array(record['tertiary']).T) / 100
            tert = np.reshape(tert, (seq_len, 9))
            evol = np.ravel(np.array(record['evolutionary']).T)
            evol = np.reshape(evol, (seq_len, num_evo_entries))

            mask = np.array(record['mask'], dtype=bool)
            dist_mat = generate_pnet_dist_matrix(
                torch.Tensor(tert), torch.Tensor(mask), mask_fill_value=-1)

            # Zero pad data
            prim_padded[:seq_len] = record['primary']
            mask_padded[:seq_len] = mask
            tert_padded[:seq_len] = tert
            evol_padded[:seq_len] = evol

            # Add data to the h3 file
            # TODO: Replace this zero padding step by directly inserting the
            # data into the datasets using the following list notation:
            # prim_set[cur_record, :seq_len] = record['primary']
            # tert_set[cur_record, :seq_len, :seq_len] = tert
            # mask_set[cur_record, :seq_len] = np.array(record['mask'], dtype=bool)
            # evol_set[cur_record, :seq_len] = evol
            id_set[cur_record] = np.string_(record['id'])
            seq_len_set[cur_record] = seq_len
            prim_set[cur_record] = prim_padded
            tert_set[cur_record] = tert_padded
            mask_set[cur_record] = mask_padded
            evol_set[cur_record] = evol_padded
            dist_set[cur_record, :seq_len, :seq_len] = dist_mat

        else:
            input_file.close()
            break

