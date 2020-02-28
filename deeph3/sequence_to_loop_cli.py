import os
import sys
from deeph3.predict import load_model
from deeph3.util import pdb2fasta, get_probs_from_model, bin_matrix, binned_dist_mat_to_values, get_dist_bins, get_omega_bins, get_theta_bins, get_phi_bins, get_bin_values, protein_dist_matrix, load_full_seq, RawTextArgumentDefaultsHelpFormatter
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
import re
import copy
import uuid
import json
from glob import glob
import torch
from tqdm import tqdm
import math
import argparse


def generate_constraints(prob_mat, pred_dist_mat, h3_range, constraint_threshold, seq, is_angle=False):
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
                prob_ji = prob_mat[j, i, binned_mat[j, i]].item()

                if prob_ij > constraint_threshold:
                    constraints.append((i, j, prob_ij, prob_mat[i, j]))
                if prob_ji > constraint_threshold:
                    constraints.append((j, i, prob_ji, prob_mat[j, i]))

    return sorted(constraints, key=lambda c: c[2], reverse=True)


def get_starting_resnum(pdb_file):
    starting_resnum = 1
    with open(pdb_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line[:4] == "ATOM":
                starting_resnum = int(line[22:26])
                break

    return starting_resnum


# def generate_constraints_from_dist(dist_mat, h3_range):
#     constraints = []
#     for i in range(h3_range[0], h3_range[1]):
#         for j in range(i):
#             constraints.append((i, j, dist_mat[i, j], 1))

#     return sorted(constraints, key=lambda c: c[2])


# def format_kic_flags(job_dir, kic_flags_template, pdb_file, weight_file=None, nstructs=500, refine=False):
#     kic_flags_file = os.path.join(job_dir, "flags")
#     output_dir = os.path.join(job_dir, "output")

#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)

#     with open(kic_flags_template, "r") as file:
#         content = file.readlines()

#     with open(kic_flags_file, "w") as file:
#         for line in content:
#             file.write(line
#                        .replace("__PDB_FILE__", pdb_file)
#                        .replace("__DIR__", job_dir)
#                        .replace("__NSTRUCTS__", str(nstructs)))


# def format_kic_script(job_dir, kic_script_template):
#     kic_script_file = os.path.join(job_dir, "kic_script.xml")
#     loop_file = os.path.join(job_dir, "loopfile")

#     with open(kic_script_template, "r") as file:
#         content = file.readlines()

#     with open(kic_script_file, "w") as file:
#         for line in content:
#             file.write(line.replace("__LOOP_FILE__", loop_file))


# def format_kic_run_con(job_dir, kic_run_con_template, con_queue=50, refine=False):
#     kic_run_con_file = os.path.join(job_dir, "kic_run.con")

#     with open(kic_run_con_template, "r") as file:
#         content = file.readlines()

#     with open(kic_run_con_file, "w") as file:
#         for line in content:
#             file.write(line
#                        .replace("__SCRIPT__", "-parser:protocol __DIR__/kic_script.xml")
#                        .replace("__DIR__", job_dir)
#                        .replace("__QUEUE__", str(con_queue)))


# def format_kic_run_slurm(job_dir, kic_run_slurm_template, refine=False):
#     kic_run_slurm_file = os.path.join(job_dir, "kic_run.slurm")

#     with open(kic_run_slurm_template, "r") as file:
#         content = file.readlines()

#     with open(kic_run_slurm_file, "w") as file:
#         for line in content:
#             file.write(line
#                        .replace("__SCRIPT__", "-parser:protocol $KIC_SCRIPT")
#                        .replace("__DIR__", job_dir))


def format_templates(job_dir, templates, formatter):
    for file, template in templates.items():
        with open(template, "r") as t:
            content = t.readlines()

        with open(os.path.join(job_dir, file), "w") as f:
            for line in content:
                for key, value in formatter.items():
                    line = line.replace(key, value)
                f.write(line)


def create_dist_constraints(hist_dir, constraint_file, constraints, seq, res_offset, constraints_weight=1):
    bin_values = get_bin_values(get_dist_bins(len(constraints[0][-1])))

    for res_i, res_j, prob, hist in constraints:
        atom1 = "CA" if seq[res_i] is "G" else "CB"
        atom2 = "CA" if seq[res_j] is "G" else "CB"

        res_i += res_offset
        res_j += res_offset

        hist_file_path = os.path.join(
            hist_dir, "dist_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * constraints_weight * round(math.log(val.item()), 5))
            # y_axis = "\t".join([str(-1 * constraints_weight * round(val.item(), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("AtomPair {} {} {} {} SPLINE dist_{}_{} {} 0 1 {}\n".format(
            atom1, res_i, atom2, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_omega_constraints(hist_dir, constraint_file, constraints, res_offset, constraints_weight=1):
    bin_values = get_bin_values(get_omega_bins(len(constraints[0][-1])))

    for res_i, res_j, prob, hist in constraints:
        res_i += res_offset
        res_j += res_offset

        hist_file_path = os.path.join(
            hist_dir, "omega_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * constraints_weight * round(math.log(val.item()), 5))
            # y_axis = "\t".join([str(-1 * constraints_weight * round(val.item(), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Dihedral CA {} CB {} CB {} CA {} SPLINE omega_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_j, res_j,
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_theta_constraints(hist_dir, constraint_file, constraints, res_offset, constraints_weight=1):
    bin_values = get_bin_values(get_theta_bins(len(constraints[0][-1])))

    for res_i, res_j, prob, hist in constraints:
        res_i += res_offset
        res_j += res_offset

        hist_file_path = os.path.join(
            hist_dir, "theta_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * constraints_weight * round(math.log(val.item()), 5))
            # y_axis = "\t".join([str(-1 * constraints_weight * round(val.item(), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Dihedral N {} CA {} CB {} CB {} SPLINE theta_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_i, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def create_phi_constraints(hist_dir, constraint_file, constraints, res_offset, constraints_weight=1):
    bin_values = get_bin_values(get_phi_bins(len(constraints[0][-1])))

    for res_i, res_j, prob, hist in constraints:
        res_i += res_offset
        res_j += res_offset

        hist_file_path = os.path.join(
            hist_dir, "phi_{}_{}".format(res_i, res_j))
        with open(hist_file_path, "w") as hist_file:
            x_axis = "\t".join([str(round(math.pi / 180 * val, 5)) for val in bin_values])
            y_axis = "\t".join([str(-1 * constraints_weight * round(math.log(val.item()), 5))
            # y_axis = "\t".join([str(-1 * constraints_weight * round(val.item(), 5))
                                for val in hist])
            hist_file.write("x_axis\t{}\n".format(x_axis))
            hist_file.write("y_axis\t{}\n".format(y_axis))

        constraint_file.write("Angle CA {} CB {} CB {} SPLINE phi_{}_{} {} 0 1 {}\n".format(
            res_i, res_i, res_j, 
            res_i, res_j, 
            hist_file_path, bin_values[1] - bin_values[0]
        ))


def write_loop_and_constraint_files(job_dir, seq, loop_params,
                     dist_constraints, omega_constraints, theta_constraints, phi_constraints,
                     constraints_weight=1, constraint_num_offset=1):
    with open(os.path.join(job_dir, "loopfile"), "w") as loop_file:
        loop_file.write("LOOP {} {} {} 0 1\n".format(
            str(loop_params[0]),
            str(loop_params[1]),
            str(loop_params[2])))

    with open(os.path.join(job_dir, "loopfile_ref"), "w") as loop_file:
        loop_file.write("LOOP {} {} {} 0 0\n".format(
            str(loop_params[0]),
            str(loop_params[1]),
            str(loop_params[2])))

    hist_dir = os.path.join(job_dir, "constraint_histograms")
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)
    
    constraint_file = os.path.join(job_dir, "constraints")
    with open(constraint_file, "w") as constraint_file:
        create_dist_constraints(
            hist_dir, constraint_file, dist_constraints, seq, constraint_num_offset,
            constraints_weight=constraints_weight)
        create_omega_constraints(
            hist_dir, constraint_file, omega_constraints, constraint_num_offset,
            constraints_weight=constraints_weight)
        create_theta_constraints(
            hist_dir, constraint_file, theta_constraints, constraint_num_offset,
            constraints_weight=constraints_weight)
        create_phi_constraints(
            hist_dir, constraint_file, phi_constraints, constraint_num_offset,
            constraints_weight=constraints_weight)
        

    return job_dir


# def create_job_dir(working_dir, constraints_num, constraint_threshold=0, weight_file=None, refine=False, native_constraints=False):
#     job_dir = os.path.join(working_dir, "top{}c_".format(
#         constraints_num) if constraints_num >= 0 else "allc_")
#     if constraint_threshold > 0:
#         job_dir += "{}p_".format(int(constraint_threshold * 100))
#     if not weight_file is None:
#         job_dir += re.split('/', weight_file)[-1] + "_"
#     if refine:
#         job_dir += "ref_"
#     if native_constraints:
#         job_dir += "native_"

#     job_dir = job_dir[:-1]
#     if os.path.exists(job_dir):
#         matches = len(list(glob(job_dir + "[0-9]*")))
#         job_dir += str(matches + 2)
#     os.mkdir(job_dir)

#     return job_dir


def _get_args():
    """Gets command line arguments"""
    predict_py_path = os.path.dirname(os.path.realpath(__file__))
    default_configfile = '/home-2/jgray21@jhu.edu/work/jgray/mywork/kic_job.json'
    default_workdir = '/home-2/jgray21@jhu.edu/work/jgray/mywork/'

    desc = (
        '''
        Convert a deeph3 prediction (from predict.py) to constraints for Rosetta loop building
        ''')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument('--configfile', type=str,
                        default=default_configfile,
                        help=('a config file'))
    parser.add_argument('--workdir', type=str,
                        default=default_workdir,
                        help=('your working directory'))
    return parser.parse_args()

def print_run_params(args):
    print("Running sequence_to_loop")
    print("     Config file : ",args.fasta_file)
    print("  Work directory : ",args.model_file, flush=True)
    return


def _cli():
    """Command line interface for %f.py when it is run as a script"""
    args = _get_args()
    for key,value in vars(args).items():
        print(key,": ",value)

    with open(args.configfile) as config_file:
        config = json.load(config_file)

    job_configs = [copy.deepcopy(config["default"])
                   for job_config in config["jobs"]]
    [job_configs[i].update(job_config)
     for i, job_config in enumerate(config["jobs"])]

    working_dir = args.workdir
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # for job_config in tqdm(job_configs):
    #     target_fasta = job_config["target_fasta"]
    #     target_pdb = job_config["target_pdb"]
    #     native_pdb = job_config["native_pdb"]
    #     checkpoint_file = job_config["model_file"]
    #     weight_file = job_config["weight_file"]
    #     nstructs = job_config["nstructs"]
    #     refine = job_config["refine"]
    #     constraint_type = job_config["constraint_type"]
    #     constraints_num = job_config["constraints_num"]
    #     constraint_threshold = job_config["constraint_threshold"]
    #     native_constraints = job_config["native_constraints"]
    #     constraints_weight = job_config["constraints_weight"]
    #     use_pdb_starting_resnum = job_config["use_pdb_starting_resnum"]
    #     con_queue = job_config["con_queue"]
    #     kic_flag_template = job_config["kic_flag_template"]
    #     kic_script_template = job_config["kic_script_template"]
    #     kic_run_con_template = job_config["kic_run_con_template"]
    #     kic_run_slurm_template = job_config["kic_run_slurm_template"]
    #     rescore_script_template = job_config["rescore_script_template"]

    #     job_dir = create_job_dir(working_dir, constraints_num, constraint_threshold,
    #                              weight_file, refine, native_constraints)
    #     with open(os.path.join(job_dir, "config.json"), "w") as job_config_file:
    #         json.dump(job_config, job_config_file)

    #     if not os.path.exists(os.path.join(job_dir, "templates")):
    #         os.mkdir(os.path.join(job_dir, "templates"))
    #     if not os.path.exists(os.path.join(job_dir, "output")):
    #         os.mkdir(os.path.join(job_dir, "output"))
    #     if not os.path.exists(os.path.join(job_dir, "con_log")):
    #         os.mkdir(os.path.join(job_dir, "con_log"))
    #     if not os.path.exists(os.path.join(job_dir, "score_pdbs")):
    #         os.mkdir(os.path.join(job_dir, "score_pdbs"))

        os.system("cp {} {}".format(target_pdb, job_dir))
        target_pdb = os.path.join(job_dir, os.path.split(target_pdb)[1])

        if target_fasta is not None:
            os.system("cp {} {}".format(target_fasta, job_dir))
            target_fasta = os.path.join(job_dir, os.path.split(target_fasta)[1])
        else:
            target_fasta = "{}.fasta".format(target_pdb[:-4])
            fasta_text = pdb2fasta(target_pdb, 2)
            # fasta_text = pdb2fasta(native_pdb, 2)
            with open(target_fasta, "w") as file:
                file.writelines(fasta_text)

        # os.system("cp {} {}".format(native_pdb, job_dir))
        # native_pdb = os.path.join(job_dir, os.path.split(native_pdb)[1])

        # os.system("cp {} {}".format(kic_flag_template, os.path.join(
        #     job_dir, "templates", "kic_flag_template")))
        # kic_flag_template = os.path.join(
        #     job_dir, "templates", "kic_flag_template")

        # os.system("cp {} {}".format(kic_script_template, os.path.join(
        #     job_dir, "templates", "kic_script_template.xml")))
        # kic_script_template = os.path.join(
        #     job_dir, "templates", "kic_script_template.xml")

        # os.system("cp {} {}".format(rescore_script_template, os.path.join(
        #     job_dir, "templates", "rescore_script_template.xml")))
        # rescore_script_template = os.path.join(
        #     job_dir, "templates", "rescore_script_template.xml")

        # if os.path.exists(kic_run_con_template):
        #     os.system("cp {} {}".format(kic_run_con_template, os.path.join(
        #         job_dir, "templates", "kic_run_template.con")))
        #     kic_run_con_template = os.path.join(
        #         job_dir, "templates", "kic_run_template.con")
        # else:
        #     kic_run_con_template = None
        # if os.path.exists(kic_run_slurm_template):
        #     os.system("cp {} {}".format(kic_run_slurm_template, os.path.join(
        #         job_dir, "templates", "kic_run_template.slurm")))
        #     kic_run_slurm_template = os.path.join(
        #         job_dir, "templates", "kic_run_template.slurm")
        # else:
        #     kic_run_slurm_template = None

        h3 = get_cdr_indices(target_pdb)['h3']
        loop_params = (h3[0], h3[1], (h3[0] + h3[1]) // 2)
        seq = load_full_seq(target_fasta)

        if native_constraints:
            # constraints = generate_constraints_from_dist(
            #     protein_dist_matrix(target_pdb), h3)
            print("\nNative constraints no longer supported\n")
            exit()
        else:
            with torch.no_grad():
                model = load_model(checkpoint_file)
                model.eval()
                probs = get_probs_from_model(
                    model, target_fasta, chain_delimiter=True)
                pred_dist_mat = binned_dist_mat_to_values(bin_matrix(probs[0], are_logits=False))

                dist_constraints = generate_constraints(probs[0], pred_dist_mat, h3, constraint_threshold, seq)
                omega_constraints = generate_constraints(probs[1], pred_dist_mat, h3, constraint_threshold, seq, is_angle=True)
                theta_constraints = generate_constraints(probs[2], pred_dist_mat, h3, constraint_threshold, seq, is_angle=True)
                phi_constraints = generate_constraints(probs[3], pred_dist_mat, h3, constraint_threshold, seq, is_angle=True)

        dist_constraints = dist_constraints[:constraints_num] if constraints_num >= 0 else dist_constraints
        omega_constraints = omega_constraints[:constraints_num] if constraints_num >= 0 else omega_constraints
        theta_constraints = theta_constraints[:constraints_num] if constraints_num >= 0 else theta_constraints
        phi_constraints = phi_constraints[:constraints_num] if constraints_num >= 0 else phi_constraints

        templates = {
            "flags": kic_flag_template,
            "kic_script.xml": kic_script_template,
            "kic_run.con": kic_run_con_template,
            "kic_run.slurm": kic_run_slurm_template,
            "rescore_script.xml": rescore_script_template
        }
        formatter = {
            "__DIR__": job_dir,
            "__QUEUE__": str(con_queue),
            "__PDB_FILE__": target_pdb,
            "__NATIVE_FILE__": native_pdb,
            "__NSTRUCTS__": str(nstructs),
            "__WEIGHTS__": "ref2015_cst" if weight_file is None else weight_file,
            "__LOOP_START__": str(loop_params[0]),
            "__LOOP_END__": str(loop_params[1])
        }

        constraint_num_offset = get_starting_resnum(native_pdb) if use_pdb_starting_resnum else 1
        write_loop_and_constraint_files(job_dir, seq, loop_params,
                         dist_constraints, omega_constraints, theta_constraints, phi_constraints,
                         constraints_weight=constraints_weight, constraint_num_offset=constraint_num_offset)
        format_templates(job_dir, templates, formatter)
        # format_kic_flags(job_dir, kic_flag_template, target_pdb, weight_file=weight_file, nstructs=nstructs, refine=refine)
        # format_kic_script(job_dir, kic_script_template)
        #
        # if not kic_run_con_template is None:
        #     format_kic_run_con(job_dir, kic_run_con_template, con_queue=con_queue, refine=refine)
        # if not kic_run_slurm_template is None:
        #     format_kic_run_slurm(job_dir, kic_run_slurm_template, refine=refine)


if __name__ == '__main__':
    _cli()
