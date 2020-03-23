import torch
import torch.nn as nn
import argparse
import pickle
import os
from deeph3.util import load_model, get_probs_from_model, bin_matrix, RawTextArgumentDefaultsHelpFormatter


def predict(model, fasta_file, chain_delimiter=True, binning_method='max',
            return_raw_probabilities=False):
    """
    Predicts the binned version of a fasta file's inter-residue distance and
    angle matrices. Alternatively, the raw probability distribution outputs
    can be returned.
    """
    probs = get_probs_from_model(model, fasta_file, chain_delimiter=chain_delimiter)
    if return_raw_probabilities:
        dist, omega, theta, phi = probs
    else:
        dist, omega, theta, phi = bin_matrix(probs, are_logits=False,
                                             method=binning_method)
    return dict(distance_matrix=dist, omega_matrix=omega,
                theta_matrix=theta, phi_matrix=phi)


def _get_args():
    """Gets command line arguments"""
    predict_py_path = os.path.dirname(os.path.realpath(__file__))
    default_model_path = os.path.join(predict_py_path, 'models/fully_trained_model.p')
    default_fasta_path = os.path.join(predict_py_path, 'data/antibody_dataset/fastas_testrun/1a0q_trunc.fasta')

    desc = (
        '''
        Outputs the logits for a given fasta file for an antibody that is structured as such:
            >[PDB ID]:H	[heavy chain sequence length]
            [heavy chain sequence]
            >[PDB ID]:L	[light chain sequence length]
            [light chain sequence]
        See 1a0q_trunc.fasta for an example.
        ''')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_file', type=str,
                        default=default_model_path,
                        help=('A pickle file containing a dictionary with the following keys:\n'
                              '    state_dict: The state dict of the H3ResNet model\n'
                              '    num_blocks1D: The number of one dimensional ResNet blocks\n'
                              '    num_blocks2D: The number of two dimensional ResNet blocks\n'
                              '    dilation (optional): The dilation cycle of the model'))
    parser.add_argument('--fasta_file', type=str,
                        default=default_fasta_path,
                        help='The fasta file used for prediction.')
    parser.add_argument('--output_raw_probabilities', type=bool,
                        default=False,
                        help=('Flag to output the raw probability distributions '
                              'rather than the binned versions'))
    parser.add_argument('--out_file', type=str,
                        default='model_out.p',
                        help='The pickle file to save the model output to.')
    parser.add_argument('--silent', type=bool,
                        default=False,
                        help='Flag to silence all run output')
    return parser.parse_args()

def print_run_params(args):
    print("Running deeph3")
    print("  Input sequence : ",args.fasta_file)
    print("           Model : ",args.model_file, flush=True)
    return

def _cli():
    """Command line interface for predict.py when it is run as a script"""
    args = _get_args()
    print_run_params(args)
    model = load_model(args.model_file)
    predictions = predict(model, args.fasta_file,
                          return_raw_probabilities=args.output_raw_probabilities)
    pickle.dump(predictions, open(args.out_file, 'wb'))


if __name__ == '__main__':
    _cli()

