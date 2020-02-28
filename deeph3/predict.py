import torch
import torch.nn as nn
import argparse
import pickle
import os
from deeph3 import H3ResNet
from deeph3.util import get_probs_from_model, bin_matrix, RawTextArgumentDefaultsHelpFormatter
from os.path import isfile


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


def load_model(file_name, num_blocks1D=3, num_blocks2D=25):
    """Loads a model from a pickle file

    :param file_name:
        A pickle file containing a dictionary with the following keys:
            state_dict: The state dict of the H3ResNet model
            num_blocks1D: The number of one dimensional ResNet blocks
            num_blocks2D: The number of two dimensional ResNet blocks
            dilation (optional): The dilation cycle of the model
    :param num_blocks1D:
        If num_blocks1D is not in the pickle file, then this number is used for
        the amount of one dimensional residual blocks.
    :param num_blocks2D:
        If num_blocks2D is not in the pickle file, then this number is used for
        the amount of two dimensional residual blocks.
    """
    if not isfile(file_name):
        raise FileNotFoundError(f'No file at {file_name}')
    checkpoint_dict = torch.load(file_name, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    dilation_cycle = 0 if not 'dilation_cycle' in checkpoint_dict else checkpoint_dict[
        'dilation_cycle']

    in_layer = list(model_state.keys())[0]
    out_layer = list(model_state.keys())[-1]
    num_out_bins = model_state[out_layer].shape[0]
    in_planes = model_state[in_layer].shape[1]

    if 'num_blocks1D' in checkpoint_dict:
        num_blocks1D = checkpoint_dict['num_blocks1D']
    if 'num_blocks2D' in checkpoint_dict:
        num_blocks2D = checkpoint_dict['num_blocks2D']

    resnet = H3ResNet(in_planes=in_planes, num_out_bins=num_out_bins,
                      num_blocks1D=num_blocks1D, num_blocks2D=num_blocks2D,
                      dilation_cycle=dilation_cycle)
    model = nn.Sequential(resnet)
    model.load_state_dict(model_state)
    model.eval()

    return model


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

