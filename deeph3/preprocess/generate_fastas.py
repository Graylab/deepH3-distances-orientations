#!/bin/python
import numpy as np
from tqdm import tqdm
from deeph3.util import get_pdb_atoms, pdb2fasta
from os import listdir, mkdir
from os.path import isdir, isfile, join


def generate_fastas(pdb_dir, out_fasta_dir, overwrite=False, num_chains=None):
    """
    Creates fasta files from residues with coordinates for all pdb files in a
    directory. Residues without coordinates will not be included in the fasta
    sequence.
    """
    if not isdir(pdb_dir):
        raise NotADirectoryError('{} is not a directory'.format(pdb_dir))
    if not isdir(out_fasta_dir):
        print('{} does not exist'.format(out_fasta_dir))
        print('Making {}...'.format(out_fasta_dir))
        mkdir(out_fasta_dir)

    # Extract pdb file names
    pdb_files = [f for f in listdir(pdb_dir) if isfile(join(pdb_dir, f)) and f[-3:] == 'pdb']

    print('Generating fasta files for {}...'.format(pdb_dir))
    for pdb_file in tqdm(pdb_files):
        fasta_file = join(out_fasta_dir, pdb_file.split('.')[0] + '.fasta')
        pdb_file = join(pdb_dir, pdb_file)
        if not isfile(fasta_file) or overwrite:
            fasta_text = pdb2fasta(pdb_file, num_chains=num_chains)
            if len(fasta_text) != 0:
                with open(fasta_file, 'w') as f:
                    f.write(fasta_text)


def seperate_fasta_by_chain(fasta_file, lines, output_dir, overwrite=False):
    """Create seperate fasta files for each heavy chain and each light chain"""
    if not isdir(output_dir):
        print('{} does not exist'.format(output_dir))
        print('Making {}...'.format(output_dir))
        mkdir(output_dir)
    file_name = fasta_file.split('.')[0]  # Remove extension
    for description, seq in [lines[:2], lines[2:]]:
        chain_type = description.split(':')[1][0]
        output_file = join(output_dir, '{}_{}.fasta'.format(file_name, chain_type))

        # Check if file already exists
        if isfile(output_file) and not overwrite:
            #print(f'{output_file} already exists')
            continue

        # Write fasta file with one chain
        with open(output_file, 'w') as new_file:
            print('Writing {}...'.format(output_file))
            new_file.write(description)
            new_file.write(seq)


def generate_H3_pdb(fasta_file, antibody_db_dir, flank=0,
                    h3_pdb_output_dir='flanked_h3_pdbs', overwrite=False):
    """
    Creates pdb file for an H3 loop plus some flank on either side of the chain

    :param fasta_file:
    :param antibody_db_dir:
    :param flank:
    :param h3_pdb_output_dir:
    :param overwrite:
    :rtype: None
    """
    if not isdir(h3_pdb_output_dir):
        print('{} does not exist'.format(h3_pdb_output_dir))
        print('Making {}...'.format(h3_pdb_output_dir))
        mkdir(h3_pdb_output_dir)

    # Create the flanked range (cdr range plus/minus a flank)
    h3_cdr_range = [95, 102]
    flanked_range = range(h3_cdr_range[0] - flank, h3_cdr_range[1] + flank + 1)
    
    # Check if the pdb exists
    pdb_file = fasta_file.split('.')[0] + '.pdb'
    pdb_file_path = join(antibody_db_dir, pdb_file)
    if not isfile(pdb_file_path):
        print('WARNING: {} does not exist, skipping file.'.format(pdb_file))
        return
    
    h3_pdb_file = join(h3_pdb_output_dir, fasta_file.split('.')[0] + '_h3.pdb')
    if not isfile(h3_pdb_file) or overwrite:
        # Read all ATOM lines
        with open(pdb_file_path, 'r') as f:
            lines = np.array([line for line in f.readlines() if 'ATOM' in line])

        try:
            data = get_pdb_atoms(pdb_file_path)
            heavy_chain_residues = data[data.chain_id == 'H']

            # Extract the residues within the h3_cdr_range
            heavy_residue_nums = heavy_chain_residues.residue_num.astype('int32')
            h3_residues = heavy_chain_residues[heavy_residue_nums.isin(flanked_range)]
            h3_residue_lines = list(h3_residues.index)

            with open(h3_pdb_file, 'w') as f:
                h3_residues = lines[h3_residue_lines]
                f.write(''.join(h3_residues))
        except IndexError:
            print("*******INDEXERROR\n file: ", pdb_file, "***********")


if __name__ == '__main__':
    generate_H3_pdb('1ors_trunc.fasta', '/home/carlos/Rosetta_REU/deep-H3-loop-prediction/deeph3/data/antibody_database',
                    flank=0, overwrite=True)

