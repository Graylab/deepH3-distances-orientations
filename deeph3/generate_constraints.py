import torch
import os
import math
import argparse
from glob import glob
from tqdm import tqdm
from deeph3.predict import load_model
from deeph3.util import load_model, get_probs_from_model, bin_matrix, binned_dist_mat_to_values, get_dist_bins, get_omega_bins, get_theta_bins, get_phi_bins, get_bin_values, load_full_seq, RawTextArgumentDefaultsHelpFormatter, get_fasta_basename


def generate_constraints(prob_mat, pred_dist_mat, h3_range, probability_threshold, seq, is_angle=False, symmetric=False):
    binvalues = get_bin_values(get_dist_bins(prob_mat.shape[2]))
    binned_mat = bin_matrix(prob_mat, are_logits=False)

    constraints = []
    for i in range(h3_range[0], h3_range[1]):
        for j in range(len(prob_mat)):
            if abs(i - j) < (1 if is_angle else 2):
                continue
            if is_angle and (seq[i] == "G" or seq[j] == "G"):
                continue

            pred_val = binvalues[binned_mat[i, j]]
            if pred_dist_mat[i, j] <= 12 and pred_val != -1:
                prob_ij = prob_mat[i, j, binned_mat[i, j]].item()
                if prob_ij > probability_threshold:
                    constraints.append((i, j, prob_ij, prob_mat[i, j]))

                prob_ji = prob_mat[j, i, binned_mat[j, i]].item()
                if not symmetric and prob_ji > probability_threshold:
                    constraints.append((j, i, prob_ji, prob_mat[j, i]))

    return sorted(constraints, key=lambda c: c[2], reverse=True)


def create_dist_constraints(hist_dir, constraint_file, constraints, seq):
    bin_values = get_bin_values(get_dist_bins(len(constraints[0][-1])))

    for i, j, prob, hist in constraints:
        # Start residue numbering at 1
        res_i = i + 1
        res_j = j + 1

        atom1 = "CA" if seq[i] is "G" else "CB"
        atom2 = "CA" if seq[j] is "G" else "CB"

        hist_file_path = os.path.join(
            hist_dir, "dist_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * round(math.log(val.item()), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("AtomPair {} {} {} {} SPLINE dist_{}_{} {} 0 1 {}\n".format(
            atom1, res_i, atom2, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_omega_constraints(hist_dir, constraint_file, constraints):
    bin_values = get_bin_values(get_omega_bins(len(constraints[0][-1])))

    for i, j, prob, hist in constraints:
        # Start residue numbering at 1
        res_i = i + 1
        res_j = j + 1

        hist_file_path = os.path.join(
            hist_dir, "omega_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * round(math.log(val.item()), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Dihedral CA {} CB {} CB {} CA {} SPLINE omega_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_j, res_j,
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_theta_constraints(hist_dir, constraint_file, constraints):
    bin_values = get_bin_values(get_theta_bins(len(constraints[0][-1])))

    for i, j, prob, hist in constraints:
        # Start residue numbering at 1
        res_i = i + 1
        res_j = j + 1

        hist_file_path = os.path.join(
            hist_dir, "theta_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * round(math.log(val.item()), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Dihedral N {} CA {} CB {} CB {} SPLINE theta_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_i, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_phi_constraints(hist_dir, constraint_file, constraints):
    bin_values = get_bin_values(get_phi_bins(len(constraints[0][-1])))

    for i, j, prob, hist in constraints:
        # Start residue numbering at 1
        res_i = i + 1
        res_j = j + 1

        hist_file_path = os.path.join(
            hist_dir, "phi_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * round(math.log(val.item()), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Angle CA {} CB {} CB {} SPLINE phi_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def write_constraint_files(output_dir, fname, seq, dist_constraints, omega_constraints, theta_constraints, phi_constraints):
    hist_dir = os.path.join(output_dir, fname+".histograms")
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)
    
    constraint_file = os.path.join(output_dir, fname+".constraints")
    with open(constraint_file, "w") as constraint_file:
        create_dist_constraints(
            hist_dir, constraint_file, dist_constraints, seq)
        create_omega_constraints(
            hist_dir, constraint_file, omega_constraints)
        create_theta_constraints(
            hist_dir, constraint_file, theta_constraints)
        create_phi_constraints(
            hist_dir, constraint_file, phi_constraints)
        

def _get_args():
    """Gets command line arguments"""
    constraint_generation_py_path = os.path.dirname(os.path.realpath(__file__))
    default_model_path = os.path.join(constraint_generation_py_path, 'models/fully_trained_model.p')
    default_fasta_path = os.path.join(constraint_generation_py_path, 'data/antibody_dataset/fastas_testrun/1a0q.fasta')

    desc = (
        '''
        Convert a deeph3 prediction (from predict.py) to constraints for Rosetta loop building
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
                        help=('The fasta file to generate constraints for.'))
    parser.add_argument('--h3',type=str,
                        default="95-102",
                        help='CDR H3 residue number range, sequentially in the fasta file (default 95-102)')
    parser.add_argument('--output_dir', type=str,
                        default='output_dir',
                        help=('Output directory for constraints'))
    parser.add_argument('--probability_threshold', type=float,
                        default=0.10,
                        help=('Minimum modal probability threshold for constraints.'))
    parser.add_argument('--topn_constraints', type=int,
                        default=0,
                        help=('Generate N most confident constraints for each geometry (0 for all).'))
    return parser.parse_args()


def print_run_params(args):
    print("Running sequence_to_loop")
    print("       Config file : ",args.fasta_file)
    print("    Work directory : ",args.model_file)
    print("  Output directory : ",args.output_dir,flush=True)
    return


def _cli():
    """Command line interface for generate_constraints.py when it is run as a script"""
    args = _get_args()
    for key,value in vars(args).items():
        print(key,": ",value)

    model_file = args.model_file
    fasta_file = args.fasta_file
    basename = get_fasta_basename(fasta_file) 
    output_dir = args.output_dir
    probability_threshold = args.probability_threshold
    topn_constraints = args.topn_constraints

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    h3str = args.h3.split('-')
    h3 = list(map(int, h3str))
    seq = load_full_seq(fasta_file)
    print("Ab sequence: ",seq)
    print("H3 sequence: ",seq[h3[0]:h3[1]])

    with torch.no_grad():
        model = load_model(model_file)
        model.eval()
        probs = get_probs_from_model(
            model, fasta_file, chain_delimiter=True)
        pred_dist_mat = binned_dist_mat_to_values(bin_matrix(probs[0], are_logits=False))

        dist_constraints = generate_constraints(probs[0], pred_dist_mat, h3, probability_threshold, seq, symmetric=True)
        omega_constraints = generate_constraints(probs[1], pred_dist_mat, h3, probability_threshold, seq, is_angle=True, symmetric=True)
        theta_constraints = generate_constraints(probs[2], pred_dist_mat, h3, probability_threshold, seq, is_angle=True)
        phi_constraints = generate_constraints(probs[3], pred_dist_mat, h3, probability_threshold, seq, is_angle=True)

    dist_constraints = dist_constraints[:topn_constraints] if topn_constraints > 0 else dist_constraints
    omega_constraints = omega_constraints[:topn_constraints] if topn_constraints > 0 else omega_constraints
    theta_constraints = theta_constraints[:topn_constraints] if topn_constraints > 0 else theta_constraints
    phi_constraints = phi_constraints[:topn_constraints] if topn_constraints > 0 else phi_constraints

    write_constraint_files(output_dir, basename, seq, dist_constraints, omega_constraints, theta_constraints, phi_constraints)


if __name__ == '__main__':
    _cli()
