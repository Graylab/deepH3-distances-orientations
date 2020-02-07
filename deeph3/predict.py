from deeph3.util import get_probs_from_model, bin_distance_matrix, get_dist_bins


def predict(model, fasta_file, chain_delimiter=True):
    dist, omega, theta, phi = get_probs_from_model(model, fasta_file, chain_delimiter=chain_delimiter)
    print(dist.shape)
    return dist, omega, theta, phi


if __name__ == '__main__':
    from deeph3 import H3ResNet
    model = H3ResNet(21, num_blocks1D=1, num_blocks2D=1)
    fasta_file = '/home/carlos/projects/deepH3-distances-orientations/deeph3/data/antibody_dataset/fastas_testrun/1a0q_trunc.fasta'
    logits = predict(model, fasta_file)

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

