import os
from glob import glob
from tqdm import tqdm
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not explicitly used, but needed for matplotlib 3d projections

from deeph3 import load_model
from deeph3.util import load_full_seq, get_probs_from_model, binned_matrix, protein_dist_matrix, protein_euler_matrix, binned_dist_mat_to_values
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from deeph3.viz import heatmap2d

from Bio import AlignIO


working_dir = "/home/jruffol1/msa_working"
fasta_dir = "deeph3/data/antibody_dataset/fastas/"
pdb_dir = "deeph3/data/antibody_dataset/pdbs_validation/"

targets = [os.path.split(t)[1][:-4] for t in list(glob(os.path.join(pdb_dir, "*.pdb")))]

# targets = targets[:5]

sequences = []
for t in targets:
    sequences.append(load_full_seq(os.path.join(fasta_dir, "{}.fasta".format(t))))

# with(open(os.path.join(working_dir, "sequences.fasta"), "w")) as msa_file:
#     for t, s in tqdm(zip(targets, sequences)):
#         msa_file.write(">{}\n{}\n".format(t, s))
# exit()

alignment = AlignIO.read(open(os.path.join(working_dir, "alignment_val.fasta")), "fasta")
# for record in alignment:
    
seq_align_maps = []
for i, s in tqdm(enumerate(sequences), total=len(sequences)):
    seq_align_maps.append([])
    seq_i = 0
    for ali_i, a in enumerate(alignment[i]):
        res = -1
        if seq_i < len(s) and s[seq_i] == a:
            res = seq_i
            seq_i += 1

        seq_align_maps[i].append(res)

# ali = alignment[37]
# seq = sequences[37]
# sam = seq_align_maps[37]
# for i, a in enumerate(ali):
#     print(i, ali[i], "\t->\t", sam[i], "" if sam[i] == -1 else seq[sam[i]])


checkpoint_file = 'deeph3/models/adam_opt_lr01_dil2/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_dil5_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'

prob_mats = {}
pickled_out_mats_file = os.path.join(working_dir, "pdb_val_out.pickle")
if os.path.exists(pickled_out_mats_file):
    with open(pickled_out_mats_file, 'rb') as f:
        print("Loading prob_mats from {}\n".format(pickled_out_mats_file))
        prob_mats = pickle.load(f)
else:
    print("Generating prob_mats from {}\n".format(checkpoint_file))
    model = load_model(checkpoint_file)
    with torch.no_grad():
        model.eval()
        for t in tqdm(targets):
            prob_mat = get_probs_from_model(model, os.path.join(fasta_dir, "{}.fasta".format(t)), chain_delimiter=True)
            prob_mats[t] = prob_mat

    with open(pickled_out_mats_file, 'wb') as f:
        pickle.dump(prob_mats, f)


real_mats = {}
pickled_real_mats_file = os.path.join(working_dir, "pdb_val_real.pickle")
if os.path.exists(pickled_real_mats_file):
    with open(pickled_real_mats_file, 'rb') as f:
        print("Loading real_mats from {}\n".format(pickled_real_mats_file))
        real_mats = pickle.load(f)
else:
    print("Generating real_mats from {}\n".format(pdb_dir))
    for t in tqdm(targets):
        real_mat = protein_dist_matrix(os.path.join(pdb_dir, "{}.pdb".format(t)))
        real_mats[t] = real_mat

    with open(pickled_real_mats_file, 'wb') as f:
        pickle.dump(real_mats, f)


dist_figs_dir = os.path.join(working_dir, "figs", "dist")
os.system("mkdir {}".format(dist_figs_dir))
prob_figs_dir = os.path.join(working_dir, "figs", "probs")
os.system("mkdir {}".format(prob_figs_dir))
acc_figs_dir = os.path.join(working_dir, "figs", "acc")
os.system("mkdir {}".format(acc_figs_dir))

aligned_dist_mat_file = os.path.join(working_dir, "aligned_dist_mat_val.pickle")
aligned_prob_mat_file = os.path.join(working_dir, "aligned_prob_mat_val.pickle")
aligned_acc_mat_file = os.path.join(working_dir, "aligned_acc_mat_val.pickle")
aligned_total_mat_file = os.path.join(working_dir, "aligned_total_mat_val.pickle")
if os.path.exists(aligned_dist_mat_file) and os.path.exists(aligned_prob_mat_file) and os.path.exists(aligned_acc_mat_file) and os.path.exists(aligned_total_mat_file):
    print("Loading aligned mats from pickles")
    with open(aligned_dist_mat_file, 'rb') as f:
        aligned_dist_mat = pickle.load(f)
    with open(aligned_prob_mat_file, 'rb') as f:
        aligned_prob_mat = pickle.load(f)
    with open(aligned_acc_mat_file, 'rb') as f:
        aligned_acc_mat = pickle.load(f)
    with open(aligned_total_mat_file, 'rb') as f:
        aligned_total_mat = pickle.load(f)
else:
    print("Generating aligned mats")
    aligned_dist_mat = np.zeros((len(alignment[0]), len(alignment[0])))
    aligned_prob_mat = np.zeros((len(alignment[0]), len(alignment[0])))
    aligned_acc_mat = np.zeros((len(alignment[0]), len(alignment[0])))
    aligned_total_mat = np.zeros((len(alignment[0]), len(alignment[0])))
    for i, t in tqdm(enumerate(targets), total=len(targets)):
        binned_mat = binned_matrix(prob_mats[t], are_logits=False)[0]
        
        dist_mat = binned_dist_mat_to_values(binned_mat)
        prob_mat = np.amax(np.asarray(prob_mats[t][0]), axis=2)
        real_mat = real_mats[t]
        
        acc_mat = np.abs(real_mat - dist_mat)
        acc_mat[np.logical_or(acc_mat <= 1, np.logical_and(dist_mat > 16, real_mat > 16.25))] = 1
        acc_mat[acc_mat > 1] = 0

        sam = seq_align_maps[i]
        for res_i in range(len(sam)):
            for res_j in range(len(sam)):
                if sam[res_i] != -1 and sam[res_j] != -1:
                    aligned_dist_mat[res_i, res_j] += dist_mat[sam[res_i], sam[res_j]]
                    aligned_prob_mat[res_i, res_j] += prob_mat[sam[res_i], sam[res_j]]
                    aligned_acc_mat[res_i, res_j] += acc_mat[sam[res_i], sam[res_j]]
                    aligned_total_mat[res_i, res_j] += 1

        heatmap2d(dist_mat, title="{} Pairwise Distance".format(t), color_min=0, color_max=16.25, out_file=os.path.join(dist_figs_dir, "{}_dist.png".format(t)))
        heatmap2d(prob_mat, title="{} Max Pairwise Probability".format(t), color_min=0, color_max=1, out_file=os.path.join(prob_figs_dir, "{}_probs.png".format(t)))
        heatmap2d(acc_mat, title="{} Max Pairwise Accuracy".format(t), color_min=0, color_max=1, out_file=os.path.join(acc_figs_dir, "{}_probs.png".format(t)))

    aligned_total_mat[aligned_total_mat == 0] = 1
    aligned_dist_mat /= aligned_total_mat
    aligned_prob_mat /= aligned_total_mat
    aligned_acc_mat /= aligned_total_mat

    with open(aligned_dist_mat_file, 'wb') as f:
        pickle.dump(aligned_dist_mat, f)
    with open(aligned_prob_mat_file, 'wb') as f:
        pickle.dump(aligned_prob_mat, f)
    with open(aligned_acc_mat_file, 'wb') as f:
        pickle.dump(aligned_acc_mat, f)
    with open(aligned_total_mat_file, 'wb') as f:
        pickle.dump(aligned_total_mat, f)

aligned_prob_mat[aligned_dist_mat > 8] = None
aligned_acc_mat[aligned_dist_mat > 8] = None
aligned_dist_mat[aligned_dist_mat > 8] = None

ali_figs_dir = os.path.join(working_dir, "figs", "ali_below8")
os.system("mkdir {}".format(ali_figs_dir))
heatmap2d(aligned_dist_mat, title="Alignment Avg Pairwise Distance".format(t), color_min=0, color_max=16.25, out_file=os.path.join(ali_figs_dir, "alignment_dist.png"))
heatmap2d(aligned_prob_mat, title="Alignment Avg Pairwise Probability".format(t), color_min=0, color_max=1, out_file=os.path.join(ali_figs_dir, "alignment_probs.png"))
heatmap2d(aligned_acc_mat, title="Alignment Avg Pairwise Accuracy".format(t), color_min=0, color_max=1, out_file=os.path.join(ali_figs_dir, "alignment_acc.png"))
heatmap2d(aligned_total_mat, title="Alignment Pairwise Totals".format(t), color_min=0, color_max=None, out_file=os.path.join(ali_figs_dir, "alignment_totals.png"))

ranges = [(0, 0.5), (0.1, 0.5), (0.1, 1), (0.25, 1), (0.5, 1), (0, 1)]
for pr in ranges:
    # sele = np.logical_not(np.logical_and(aligned_prob_mat > pr[0], aligned_dist_mat <= 8))
    sele = np.logical_not(np.logical_and(aligned_prob_mat >= pr[0], aligned_prob_mat <= pr[1]))
    cp_aligned_prob_mat = np.copy(aligned_prob_mat)
    cp_aligned_acc_mat = np.copy(aligned_acc_mat)
    cp_aligned_prob_mat[sele] = None
    cp_aligned_acc_mat[sele] = None


    # heatmap2d(cp_aligned_prob_mat, title="Alignment Avg Max Pairwise Probability".format(t), color_min=pr[0], color_max=pr[1], out_file=os.path.join(prob_figs_dir, "alignment_probs_{}-{}_d8.png".format(pr[0], pr[1])))
    # heatmap2d(cp_aligned_acc_mat, title="Alignment Avg Pairwise Accuracy".format(t), color_min=0, color_max=1, out_file=os.path.join(acc_figs_dir, "alignment_acc_{}-{}_d8.png".format(pr[0], pr[1])))

    heatmap2d(cp_aligned_prob_mat, title="Alignment Avg Max Pairwise Probability".format(t), color_min=pr[0], color_max=pr[1], out_file=os.path.join(ali_figs_dir, "alignment_probs_{}-{}p.png".format(pr[0], pr[1])))
    heatmap2d(cp_aligned_acc_mat, title="Alignment Avg Pairwise Accuracy".format(t), out_file=os.path.join(ali_figs_dir, "alignment_acc_{}-{}p.png".format(pr[0], pr[1])))

for ar in ranges:
    # sele = np.logical_not(np.logical_and(aligned_prob_mat > pr[0], aligned_dist_mat <= 8))
    sele = np.logical_not(np.logical_and(aligned_acc_mat >= ar[0], aligned_acc_mat <= ar[1]))
    cp_aligned_prob_mat = np.copy(aligned_prob_mat)
    cp_aligned_acc_mat = np.copy(aligned_acc_mat)
    cp_aligned_prob_mat[sele] = None
    cp_aligned_acc_mat[sele] = None


    # heatmap2d(cp_aligned_prob_mat, title="Alignment Avg Max Pairwise Probability".format(t), color_min=pr[0], color_max=pr[1], out_file=os.path.join(prob_figs_dir, "alignment_probs_{}-{}_d8.png".format(pr[0], pr[1])))
    # heatmap2d(cp_aligned_acc_mat, title="Alignment Avg Pairwise Accuracy".format(t), color_min=0, color_max=1, out_file=os.path.join(acc_figs_dir, "alignment_acc_{}-{}_d8.png".format(pr[0], pr[1])))

    heatmap2d(cp_aligned_prob_mat, title="Alignment Avg Max Pairwise Probability".format(t), out_file=os.path.join(ali_figs_dir, "alignment_probs_{}-{}a.png".format(ar[0], ar[1])))
    heatmap2d(cp_aligned_acc_mat, title="Alignment Avg Pairwise Accuracy".format(t), color_min=ar[0], color_max=ar[1], out_file=os.path.join(ali_figs_dir, "alignment_acc_{}-{}a.png".format(ar[0], ar[1])))

