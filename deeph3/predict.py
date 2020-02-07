import torch
from deeph3.util import get_probs_from_model, bin_matrix, bin_distance_matrix, get_dist_bins


def predict(model, fasta_file, chain_delimiter=True, binning_method='max'):
    """
    Predicts the binned version of a distance matrix
    """
    probs = get_probs_from_model(model, fasta_file, chain_delimiter=chain_delimiter)
    dist, omega, theta, phi = bin_matrix(probs, are_logits=False,
                                         method=binning_method)
    return dict(distance_matrix=dist, omega_matrix=omega,
                theta_matrix=theta, phi_matrix=phi)


if __name__ == '__main__':
    from deeph3 import H3ResNet
    model = H3ResNet(21, num_blocks1D=1, num_blocks2D=1)
    fasta_file = '/home/carlos/projects/deepH3-distances-orientations/deeph3/data/antibody_dataset/fastas_testrun/1a0q_trunc.fasta'
    print(predict(model, fasta_file))

    """
    import argparse
    from deeph3 import load_model

    desc = (
        '''
        Outputs the logits for a given fasta file for an antibody that is structured as such:
        >[PDB ID]:H	[heavy chain sequence length]
        [heavy chain sequence]
        >[PDB ID]:L	[light chain sequence length]
        [light chain sequence]
        
        See 1a0q_trunc.fasta for an example.
        ''')
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('data_dir', type=str,
                        help='The directory containing the training and validation '
                             'h5 files.')
    """

