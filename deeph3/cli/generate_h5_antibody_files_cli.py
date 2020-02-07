import argparse
import os.path
from deeph3.preprocess.generate_h5_antibody_files import antibody_to_h5
from deeph3.util import split_dir


desc = 'Creates h5 files from all the pdb files in a directory'
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
remove_missing_n_term = args.remove_missing_n_term
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
        set_type = os.path.split(sub_dir)[1].split('_')[-1]
        h5_file = '{}_{}.h5'.format(os.path.splitext(out_file)[0], set_type)
        print('Generating h5 file for {}'.format(h5_file))
        antibody_to_h5(sub_dir, h5_file, fasta_dir=fasta_dir,
                       overwrite=overwrite, print_progress=True,
                       remove_missing_n_term=remove_missing_n_term)
else:
    antibody_to_h5(pdb_dir, out_file, fasta_dir=fasta_dir, overwrite=overwrite,
                   print_progress=True, remove_missing_n_term=remove_missing_n_term)

