import pickle
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from deeph3.util import protein_dist_matrix


def generate_all_dist_matrices(pdb_dir, out_dir):
    """Gets the distance matrices of all the PDBs in a directory"""
    # Extract pdb file names
    pdb_files = [f for f in listdir(pdb_dir) if isfile(join(pdb_dir, f)) and f[-3:] == 'pdb']

    print('Generating distance matrices (CB to CB)...')
    for pdb_file in tqdm(pdb_files):
        protein_name = pdb_file.split('.')[0]
        dist_file = join(out_dir, '{}.p'.format(protein_name))
        if not isfile(dist_file):
            pdb_file_path = join(pdb_dir, pdb_file)
            pickle.dump(protein_dist_matrix(pdb_file_path), open(dist_file, 'wb'))


if __name__ == '__main__':
    def main():
        from deeph3.viz import heatmap2d
        from deeph3.util import load_full_seq
        from Bio.PDB import PDBParser
        import numpy as np
        import re

        fasta_file = '../data/fastas/1gig_trunc.fasta'
        pdb_file = '../data/antibody_database_test/1gig_trunc.pdb'
        with open(pdb_file, 'r') as f:
            lines = f.readlines()

        lines = [line for line in lines if 'ATOM' in line]
        # Split PDB by lines (rows) and spaces (columns)
        data = np.array([re.sub('\s+', ' ', line).split(' ')[:9] for line in lines])
        heavy_chain_residues = data[data[:, 4] == 'H']
        light_chain_residues = data[data[:, 4] == 'L']
        heavy_residue_set = set()
        for res_id in heavy_chain_residues[:, 5]:
            res_num = re.sub('\D', '', res_id)
            if len(res_num) == 1:
                heavy_residue_set.add('00' + res_id)
            elif len(res_num) == 2:
                heavy_residue_set.add('0' + res_id)
            else:
                heavy_residue_set.add(res_id)
        heavy_residue_set = sorted(list(heavy_residue_set))

        light_residue_set = set()
        for res_id in light_chain_residues[:, 5]:
            res_num = re.sub('\D', '', res_id)
            if len(res_num) == 1:
                light_residue_set.add('00' + res_id)
            elif len(res_num) == 2:
                light_residue_set.add('0' + res_id)
            else:
                light_residue_set.add(res_id)
        light_residue_set = sorted(list(light_residue_set))

        print('PDB alone')
        print(len(heavy_residue_set) + len(light_residue_set))
        dist_mat = protein_dist_matrix(pdb_file)
        seq = load_full_seq(fasta_file)
        print(len(seq))

        from Bio.SeqUtils import seq1
        parser = PDBParser()
        structure = parser.get_structure('1gig', pdb_file)
        for chain in structure.get_chains():
            id_ = chain.id
            seq = ''.join([residue.resname for residue in chain])
            print(id_)
            print(seq)
            print(seq1(seq))
            print(len(seq1(seq)))

        from deeph3.preprocess.antibody_text_parser import get_info
        print(get_info(pdb_file))

    main()
