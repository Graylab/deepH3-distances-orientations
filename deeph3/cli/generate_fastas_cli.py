import sys
from tqdm import tqdm
from os import listdir, chdir
from os.path import isdir, isfile, join, abspath, dirname
from pathlib import Path
from deeph3.preprocess.generate_fastas import seperate_fasta_by_chain, generate_fastas, generate_H3_pdb


if len(sys.argv) != 2:
    print('USAGE:')
    print('python {} truncated_antibody_database_dir'.format(sys.argv[0]))
    print('truncated_antibody_database_dir can only contain antibody pdb\'s with the')
    print('heavy (H) and light (L) chains')
    sys.exit()

antibody_db_dir = sys.argv[1]
if not isdir(antibody_db_dir):
    print('ERROR: {} is not a directory'.format(antibody_db_dir))
    sys.exit()
antibody_db_dir = abspath(antibody_db_dir)

antibody_data = str(Path(dirname(abspath(__file__))).parent)
antibody_data = str(Path('{}/data'.format(antibody_data)))
chdir(antibody_data)

fasta_dir = 'fastas'
full_chain_output_dir = 'single_chain_fastas'
generate_fastas(antibody_db_dir, fasta_dir, num_chains=2)

# Extract fasta file names
fasta_files = [f for f in listdir(fasta_dir) if isfile(join(fasta_dir, f)) and f[-5:] == 'fasta']

print('Generating heavy/light chain fasta files...')
for fasta_file in tqdm(fasta_files):
    with open(join(fasta_dir, fasta_file), 'r') as f:
        lines = f.readlines()
        if len(lines) != 4:
            print('ERROR: skipping {}. Expected 4 lines in the file, got {}'.format(f, len(lines)))
            continue
        seperate_fasta_by_chain(fasta_file, lines, full_chain_output_dir)

print('Generating flanked H3 loop fasta files...')
h3_pdb_output_dir = 'flanked_h3_pdbs'
h3_fasta_output_dir = 'flanked_h3_fastas'
for fasta_file in tqdm(fasta_files):
    with open(join(fasta_dir, fasta_file), 'r') as f:
        generate_H3_pdb(fasta_file, antibody_db_dir, full_chain_output_dir, h3_pdb_output_dir=h3_pdb_output_dir)
generate_fastas(h3_pdb_output_dir, h3_fasta_output_dir, num_chains=1)

