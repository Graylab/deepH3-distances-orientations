import h5py
import numpy as np
import argparse
from tqdm import tqdm
from deeph3.preprocess import antibody_text_parser as parser
from deeph3.util import split_dir
from os import listdir, remove
import os.path as path


def antibody_to_h5(pdb_dir, out_file_path, fasta_dir=None,
                   overwrite=False, print_progress=False):
    pdb_files = [_ for _ in listdir(pdb_dir) if _[-3:] == 'pdb']
    num_seqs = len(pdb_files)

    if fasta_dir is not None:
        seq_info = parser.antibody_db_seq_info(fasta_dir)
        max_h_len = seq_info['max_heavy_seq_len']
        max_l_len = seq_info['max_light_seq_len']
        max_total_len = seq_info['max_total_seq_len']
    else:
        print('WARNING: No fasta directory given! Defaulting max sequence '
              'length for both heavy/lights chains to 300')
        max_h_len = 300
        max_l_len = 300
        max_total_len = 600

    if overwrite and path.isfile(out_file_path):
        remove(out_file_path)
    h5_out = h5py.File(out_file_path, 'w')
    id_set = h5_out.create_dataset('id', (num_seqs,),
                                   compression='lzf', dtype='S25',
                                   maxshape=(None,))
    h_len_set = h5_out.create_dataset('heavy_chain_seq_len',
                                      (num_seqs,),
                                      compression='lzf', dtype='uint16',
                                      maxshape=(None,),
                                      fillvalue=0)
    l_len_set = h5_out.create_dataset('light_chain_seq_len',
                                      (num_seqs,),
                                      compression='lzf', dtype='uint16',
                                      maxshape=(None,),
                                      fillvalue=0)
    h_prim_set = h5_out.create_dataset('heavy_chain_primary',
                                       (num_seqs, max_h_len),
                                       compression='lzf', dtype='uint8',
                                       maxshape=(None, max_h_len),
                                       fillvalue=-1)
    l_prim_set = h5_out.create_dataset('light_chain_primary',
                                       (num_seqs, max_l_len),
                                       compression='lzf', dtype='uint8',
                                       maxshape=(None, max_l_len),
                                       fillvalue=-1)
    h1_set = h5_out.create_dataset('h1_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    h2_set = h5_out.create_dataset('h2_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    h3_set = h5_out.create_dataset('h3_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    l1_set = h5_out.create_dataset('l1_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    l2_set = h5_out.create_dataset('l2_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    l3_set = h5_out.create_dataset('l3_range', (num_seqs, 2),
                                   compression='lzf', dtype='uint16',
                                   fillvalue=-1)
    dist_angle_set = h5_out.create_dataset('dist_angle_mat',
                                           (num_seqs, 4, max_total_len,
                                            max_total_len),
                                           maxshape=(
                                               None, 4, max_total_len, max_total_len),
                                           compression='lzf', dtype='float',
                                           fillvalue=-1)

    for index, file in tqdm(enumerate(pdb_files), disable=(not print_progress), total=len(pdb_files)):
        # Get all file names
        id_ = parser.get_id(file)
        pdb_file = str(path.join(pdb_dir, id_ + '.pdb'))

        fasta_file = None if fasta_dir is None else str(
            path.join(fasta_dir, id_ + '.fasta'))
        info = parser.get_info(pdb_file, fasta_file=fasta_file,
                               verbose=False)

        # Get primary structures
        heavy_prim = info['H']
        light_prim = info['L']

        total_len = len(heavy_prim) + len(light_prim)

        id_set[index] = np.string_(id_)

        try:
            dist_angle_set[index, :4, :total_len,
                           :total_len] = np.array(info['dist_angle_mat'])
        except TypeError:
            msg = ('Fasta/PDB coordinate length mismatch: the fasta sequence '
                   'length of {} and the number of coordinates ({}) in {} '
                   'mismatch.\n ')
            raise ValueError(msg.format(total_len, len(
                info['dist_angle_mat']), pdb_file))

        h_len_set[index] = len(heavy_prim)
        l_len_set[index] = len(light_prim)

        h_prim_set[index, :len(heavy_prim)] = np.array(heavy_prim)
        l_prim_set[index, :len(light_prim)] = np.array(light_prim)

        for h_set, name in [(h1_set, 'h1'), (h2_set, 'h2'), (h3_set, 'h3'),
                            (l1_set, 'l1'), (l2_set, 'l2'), (l3_set, 'l3')]:
            if len(info[name]) == 1:
                info[name] = [info[name][0], info[name][0]]
            # Skip loops that do not have residues
            if len(info[name]) == 2:
                h_set[index] = np.array(info[name])
            else:
                print(info[name])
                msg = 'WARNING: {} does not have any coordinates for the {} loop!'
                print(msg.format(file, name))


def cli():
    desc = 'Creates h5 files from all the ProteinNet text files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('pdb_dir', type=str,
                        help='The directory containing PDB files where an antibody'
                             'with a PDB id of ID is named: ID.pdb')
    parser.add_argument('fasta_dir', type=str,
                        help='The directory containing fastas files where an '
                             'antibody with a PDB id of ID is named: ID.fasta')
    parser.add_argument('out_file', type=str,
                        help='The name of the outputted h5 file. This should be a '
                             'absolute path, otherwise it is output to the '
                             'working directory.')
    parser.add_argument('--split', type=bool, default=False,
                        help='Whether or not to randomly split the pdb dir into '
                             'seperate train, and validation directories and '
                             'make seperate h5 files for each.')
    parser.add_argument('--get_seq_from_pdb_file', type=bool, default=False,
                        help='Flag to ignore the fasta_dir argument and derive the '
                             'sequence from the PDB file\'s residues. WARNING: '
                             'setting this flag assumes that the PDB file has '
                             'coordinates for every residue in its sequence. This '
                             'is not typically the case.')
    parser.add_argument('--remove_missing_n_term', type=bool, default=True)
    parser.add_argument('--train_set_proportion', type=float, default=0.95)
    parser.add_argument('--validation_set_proportion', type=float, default=0.05)
    parser.add_argument('--overwrite', type=bool,
                        help='Whether or not to overwrite a file or not,'
                             ' if it exists',
                        default=False)

    args = parser.parse_args()
    pdb_dir = args.pdb_dir
    fasta_dir = None if args.get_seq_from_pdb_file else args.fasta_dir
    out_file = args.out_file

    overwrite = args.overwrite
    split = args.split

    if split:
        train_set_prop = args.train_set_proportion
        val_set_prop = args.validation_set_proportion
        props = [train_set_prop, val_set_prop]

        base = '{}_'.format(pdb_dir) + '{}'
        dir_names = [base.format(_) for _ in ['train', 'validation']]
        props = [_ for _ in props if _ != 0]
        dir_names = [_ for _, p in zip(dir_names, props) if p != 0]
        split_dir(pdb_dir, props, dir_names=dir_names, print_progress=True)

        for sub_dir in dir_names:
            set_type = path.split(sub_dir)[1].split('_')[-1]
            h5_file = '{}_{}.h5'.format(path.splitext(out_file)[0], set_type)
            print('Generating h5 file for {}'.format(h5_file))
            antibody_to_h5(sub_dir, h5_file, fasta_dir=fasta_dir,
                           overwrite=overwrite, print_progress=True)
    else:
        antibody_to_h5(pdb_dir, out_file, fasta_dir=fasta_dir, overwrite=overwrite,
                       print_progress=True)


if __name__ == '__main__':
    cli()

