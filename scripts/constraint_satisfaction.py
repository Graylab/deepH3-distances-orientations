from deeph3 import load_model
from deeph3.util import pdb2fasta, get_probs_from_model, binned_matrix, get_dist_bins, get_omega_bins, get_theta_bins, get_phi_bins, get_bin_values, protein_dist_matrix, protein_dist_angle_matrix, binned_dist_mat_to_values, load_full_seq
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from deeph3.viz import heatmap2d
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import circcorrcoef
from glob import glob
import torch


# params = {
#     'savefig.dpi': 200,
#     'legend.fontsize': '24',
#     'axes.labelsize': '20',
#     'axes.titlesize':'20',
#     'xtick.labelsize':'18',
#     'ytick.labelsize':'18'
# }
# plt.rcParams.update(params)


def make_constraint(i, j, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=999):
    prob = prob_mat[i, j, binned_mat[i, j]].item()
    bin = binned_mat[i, j]
    mean = binvalues[bin]
    # if bin < len(binvalues) - 2 and prob > 0:
    if pred_dist <= 12 and prob > 0:
    # if prob > 0:
        stdev = torch.sqrt(torch.sum(
            torch.FloatTensor([prob_mat[i, j, k] * pow(binvalues[k] - mean, 2) for k in range(len(binvalues))])))
        real_val = real_mat[i][j].item()
        # real_val = real_val if real_val <= binvalues[-1] else binvalues[-1]
        constraints.append((binvalues[binned_mat[i, j]], real_val, prob, stdev))


def get_all_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins):
    binvalues = get_bin_values(bins)
    binned_mat = binned_matrix(prob_mat, are_logits=False)
    h3 = cdr['h3']
    constraints = []
    if len(h3) < 2: return constraints

    for i in range(h3[0], h3[1]):
    # for i in range((h3[0] + h3[1])//2 - 1, (h3[0] + h3[1])//2 + 1):
        # for j in range(i):
        for j in range(len(prob_mat)):
            if abs(i - j) < 1 or real_mat[i, j] == -1:
                continue
            make_constraint(i, j, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[i, j])
            make_constraint(j, i, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[j, i])

    constraints = sorted(constraints, key=lambda c: c[2], reverse=True)
    return constraints


def get_loop_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins):
    binvalues = get_bin_values(bins)
    binned_mat = binned_matrix(prob_mat, are_logits=False)

    h3 = cdr['h3']
    constraints = []
    if len(h3) < 2: return constraints

    for i in range(h3[0], h3[1]):
    # for i in range((h3[0] + h3[1])//2 - 1, (h3[0] + h3[1])//2 + 1):
        # for j in range(i):
        for j in range(len(prob_mat)):
            if not (j in range(*cdr['h1']) or
                    j in range(*cdr['h2']) or
                    j in range(*cdr['h3']) or
                    j in range(*cdr['l1']) or
                    j in range(*cdr['l2']) or
                    j in range(*cdr['l3'])):
                continue

            # if not (j in range(*cdr['h3'])):
            #     continue
            if abs(i - j) < 1 or real_mat[i, j] == -1:
                continue
            make_constraint(i, j, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[i, j])
            make_constraint(j, i, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[j, i])

    constraints = sorted(constraints, key=lambda c: c[2], reverse=True)
    return constraints


def get_h3_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins):
    binvalues = get_bin_values(bins)
    binned_mat = binned_matrix(prob_mat, are_logits=False)

    h3 = cdr['h3']
    constraints = []
    if len(h3) < 2: return constraints

    for i in range(h3[0], h3[1]):
    # for i in range((h3[0] + h3[1])//2 - 1, (h3[0] + h3[1])//2 + 1):
        for j in range(h3[0], h3[1]):
            if abs(i - j) < 1 or real_mat[i, j] == -1:
                continue
            make_constraint(i, j, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[i, j])
            make_constraint(j, i, constraints, prob_mat, real_mat, binned_mat, binvalues, pred_dist=dist_mat[j, i])

    constraints = sorted(constraints, key=lambda c: c[2], reverse=True)
    return constraints


def constraints_within_tol(constraints, prob_threshold, tol):
    within_tol = []
    outside_tol = []

    for (dist, pred, prob, stdev) in constraints:
        tol = stdev
        if prob < prob_threshold:
            continue
        if dist + tol > pred > dist - tol:
            within_tol.append([dist, pred, prob, stdev])
        else:
            outside_tol.append([dist, pred, prob, stdev])

    return np.asarray(within_tol), np.asarray(outside_tol)


# checkpoint_file = 'deeph3/models/adam_opt_lr01_run2/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'
# checkpoint_file = 'deeph3/models/adam_opt_lr01/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234_epoch18_batch347.p'
checkpoint_file = 'deeph3/models/adam_opt_lr01_da/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_bins26_dil5_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'


# target_fastas = ["/home/jruffol1/kic_working/mut1/mut1.fasta", "/home/jruffol1/kic_working/mut2/mut2.fasta",
#                  "/home/jruffol1/kic_working/mut3/mut3.fasta", "/home/jruffol1/kic_working/mut4/mut4.fasta"]
# target_pdbs = ["/home/jruffol1/kic_working/mut1/mut1.pdb", "/home/jruffol1/kic_working/mut2/mut2.pdb",
#                "/home/jruffol1/kic_working/mut3/mut3.pdb", "/home/jruffol1/kic_working/mut4/mut4.pdb"]

# target_pdbs = ["/home/jruffol1/kic_working/mut1/mut1.pdb"]

# target_fastas = sorted(list(glob("/home/jruffol1/kic_working/antibody_benchmark_rescore/experiment_benchmark/**/*.fasta")))
target_pdbs = sorted(list(glob("/home/jruffol1/kic_working/antibody_benchmark_rescore/benchmark/**/natives/*.truncated_0001.pdb")))

# target_fastas = sorted(list(glob(
#     "/home/cguerra3/rosetta_REU/deep-H3-loop-prediction/deeph3/data/fastas/*.fasta")))[:40]
# target_pdbs = sorted(list(glob(
#     "/home/cguerra3/rosetta_REU/deep-H3-loop-prediction/deeph3/data/antibody_database/*.pdb")))[:40]


parent_out_dir = "/home/jruffol1/kic_working/const_satis"
if not os.path.exists(parent_out_dir):
    os.mkdir(parent_out_dir)


target_fastas = []
fasta_dir = os.path.join(parent_out_dir, "fastas")
os.system("mkdir {}".format(fasta_dir))
for target_pdb in target_pdbs:
    target_fasta = os.path.join(fasta_dir, "{}.fasta".format(os.path.split(target_pdb)[1][:-4]))
    fasta_text = pdb2fasta(target_pdb, 2)
    with open(target_fasta, "w") as file:
        file.writelines(fasta_text)
    
    target_fastas.append(target_fasta)

num_bins = 26
bins_list = [get_dist_bins(num_bins), get_omega_bins(num_bins), get_theta_bins(num_bins), get_phi_bins(num_bins)]

output_names = ["dist", "omega", "theta", "phi"]
all_constraints = [[] for _ in range(len(output_names))]

for t, (target_fasta, target_pdb) in tqdm(list(enumerate(zip(target_fastas, target_pdbs)))):
    model = load_model(checkpoint_file)
    with torch.no_grad():
        model.eval() 
        # print(target_fasta)

        prob_mats = get_probs_from_model(model, target_fasta, chain_delimiter=True)
        real_mats = protein_dist_angle_matrix(target_pdb)
        dist_mat = binned_dist_mat_to_values(binned_matrix(prob_mats[0], are_logits=False))

        for n, name in enumerate(output_names):
            out_dir = os.path.join(parent_out_dir, name)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            prob_mat = prob_mats[n]
            real_mat = real_mats[n]
            cdr = get_cdr_indices(pdb_file_path=target_pdb)

            # constraints = get_all_h3_constraints(cdr, prob_mat, real_mat)
            # constraints = get_loop_h3_constraints(cdr, prob_mat, real_mat)

            target = os.path.split(target_pdb)[1][:-4]
            # prob_threshs = [0, 0.1, 0.15, 0.2, 0.25]
            prob_threshs = [0, 0.1, 0.2]
            tol = 0.5

            # plt.figure(figsize=(20, 12))
            plt.figure(figsize=(12, 12))
            plt.title("H3 Distances")
            plt.xlabel("Predicted Distance (A)")
            plt.ylabel("True Distance (A)")
            projection = None# if n == 0 else 'polar'


            bins = bins_list[n]
            max_val = 12 if n == 0 else 999
            constraints_list = [get_all_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins),
                                get_loop_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins),
                                get_h3_h3_constraints(cdr, prob_mat, real_mat, dist_mat, bins)]
            all_constraints[n] += constraints_list[0]
            for c_loop_i, constraints in enumerate(constraints_list):
                for i, prob in enumerate(prob_threshs):
                    within_tol, outside_tol = constraints_within_tol(
                        constraints, prob, tol)
                    plt.subplot(len(constraints_list), len(prob_threshs),
                                c_loop_i * len(prob_threshs) + i + 1, projection=projection)
                    plt.title("Prob > {}% ({}/{})".format(prob * 100,
                                                        len(within_tol), len(within_tol) + len(outside_tol)))
                    plt.xlabel("Predicted {}".format(name))
                    plt.ylabel("True {}".format(name))
                    
                    plot_ranges = [(0, 20), (-180, 180), (-180, 180), (0, 180)]
                    plt.xlim(plot_ranges[n])
                    plt.ylim(plot_ranges[n])
                    plt.plot(plot_ranges[n], plot_ranges[n], 'k-')

                    if len(within_tol):
                        plt.errorbar(within_tol[:, 0], within_tol[:, 1], yerr=within_tol[:, 3],
                                    c='b', ls='none', marker='s', mfc='blue')
                    if len(outside_tol):
                        plt.errorbar(outside_tol[:, 0], outside_tol[:, 1], yerr=outside_tol[:, 3],
                                    c='r', ls='none', marker='s', mfc='red')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "{}_conf.pdf".format(target)))
            plt.close()
            # print("Saved: ", os.path.join(out_dir, "{}_conf.pdf".format(target)))

            # for c_loop_i, constraints in enumerate(constraints_list):
            #     plt.figure(figsize=(5, 5))
            #     plt.title("All H3 Distances" if c_loop_i ==
            #               0 else "Interloop H3 Distances")
            #     plt.xlabel("Confidence (A)")
            #     plt.ylabel("Accuracy (A)")
            #     plt.xlim((0, 0.32))
            #     plt.ylim((-0.1, 1.1))
            #     plt.plot((-1, 2), (-1, 2), 'k-')
            #     prob_threshs = np.linspace(0, 0.3, 11)
            #     data = np.zeros(shape=(len(prob_threshs) - 1, 3))
            #     for i in range(1, len(prob_threshs)):
            #         d = []
            #         for (dist, pred, prob, stdev) in constraints:
            #             tol = stdev
            #             if prob < prob_threshs[i - 1] or prob > prob_threshs[i]:
            #                 continue
            #             d.append(1) if dist + tol > pred > dist - tol else d.append(0)
            #         data[i - 1][0] = prob_threshs[i]
            #         data[i - 1][1] = np.mean(d)
            #         data[i - 1][2] = len(d)

            #     plt.scatter(data.T[0], data.T[1])
            #     for i in range(len(data)):
            #         plt.annotate(int(data[i][2]), (data[i][0], data[i][1]))
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(out_dir, "{}_{}_pred_conf.pdf".format(
            #         "all" if c_loop_i == 0 else "loop", target)))
            #     plt.close()


output_names = ["d (Å)", "ω (deg)", "θ (deg)", "φ (deg)"]
plot_ranges = [(4, 16), (-180, 180), (-180, 180), (0, 180)]
axis_ticks = [[4, 8, 12, 16], [-180, -90, 0, 90, 180], [-180, -90, 0, 90, 180], [0, 45, 90, 135, 180]]
for const_thresh in [0, 0.1]:
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("True Value")
    for i, (name, constraints) in enumerate(zip(output_names, all_constraints)):
        thresh_constraints = []
        for c in constraints:
            if i == 0 and (c[1] > bins_list[0][-1][0] or c[1] < bins_list[0][0][1]):
                continue
            if c[2] > const_thresh:
                thresh_constraints.append((c[0], c[1]))
        thresh_constraints = np.asarray(thresh_constraints).T

        if i == 0 or i == 3:
            print("{} corr ({}): {}".format(name, const_thresh, np.corrcoef(thresh_constraints)[0, 1]))
        else:
            print("{} corr ({}): {}".format(name, const_thresh, circcorrcoef(thresh_constraints[0] *np.pi/180, thresh_constraints[1]*np.pi/180)))
        
        out_dir = os.path.join(parent_out_dir, name)
        plt.subplot(2, 2, i+1)
        plt.hist2d(thresh_constraints[0], thresh_constraints[1], 
                bins=(24 if i == 0 else 26), range=(plot_ranges[i], plot_ranges[i]),
                cmap='YlOrRd', cmin=1)
        plt.title(name)
        plt.xticks(axis_ticks[i])
        plt.yticks(axis_ticks[i])

    plt.tight_layout()
    plt.savefig(os.path.join(parent_out_dir, "heatmap_{}.pdf".format(const_thresh)))
    plt.close()