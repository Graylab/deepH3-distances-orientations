import sys
import argparse
from os import mkdir
from os.path import abspath, isdir
from deeph3.preprocess.generate_antibody_db_dist_matrices import generate_all_dist_matrices


desc = ('Creates pickled distance matrices for all the antibody PDB files in a '
        'directory')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('pdb_dir', type=str,
                    helo='The directory with the antibody PDB files')
parser.add_argument('--out_dir', type=str,
                    help='The directory to output the distance matrices to',
                    default='dists')

args = parser.parse_args()
pdb_dir = args.pdb_dir
dist_dir = args.out_dir

if not isdir(dist_dir):
    print('{} not found. Making {}...'.format(dist_dir, dist_dir))
    mkdir(dist_dir)

if not isdir(pdb_dir):
    print('ERROR: {} is not a directory'.format(pdb_dir))
    sys.exit(-1)

pdb_dir = abspath(pdb_dir)
generate_all_dist_matrices(pdb_dir, dist_dir)

