from deeph3 import load_model
from deeph3.util import get_probs_from_model, binned_matrix, get_dist_bins, protein_dist_matrix, load_full_seq
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from Bio.PDB import PDBParser
import numpy as np
import torch


target_name = "mut4"
target_pdb = "/home/jruffol1/kic_working/mut4/mut4.pdb"
target_fasta = "/home/jruffol1/kic_working/mut4/mut4.fasta"

checkpoint_file = 'deeph3/models/adam_opt_lr01_dil2/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_dil5_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'
model = load_model(checkpoint_file)
prob_mat = get_probs_from_model(model, target_fasta, chain_delimiter=True)
print(prob_mat.shape)
prob_mat = prob_mat[0]
dist_mat = protein_dist_matrix(target_pdb)

cdr = get_cdr_indices(target_pdb)
h3 = get_cdr_indices(target_pdb)['h3']
loop_params = (h3[0], h3[1], (h3[0] + h3[1]) // 2)
seq = load_full_seq(target_fasta)

binvalues = [t[0] for t in get_dist_bins(prob_mat.shape[2])]
binvalues = [v + (binvalues[2] - binvalues[1]) / 2 for v in binvalues]
binned_mat = binned_matrix(prob_mat, are_logits=False)

constraint_threshold = 0.15
constraints = []
for i in range(h3[0], h3[1]):
    for j in range(i):
        if abs(i - j) < 2:
            continue
        prob = prob_mat[i, j, binned_mat[i,j]].item()
        bin = binned_mat[i, j]
        mean = binvalues[bin]
        # if bin < len(binvalues) - 2 and prob > 0:
        if mean <= 12 and prob > constraint_threshold:
            stdev = max(0.25, torch.sqrt(torch.sum(
                torch.FloatTensor([prob_mat[i, j, k] * pow(binvalues[k] - mean, 2) for k in range(len(binvalues))]))))
            constraints.append((i, j, prob, binvalues[binned_mat[i, j]], dist_mat[i][j].item(), stdev))

constraints = sorted(constraints, key=lambda c: c[2], reverse=True)
# constraints = sorted(constraints, key=lambda c: min(abs(c[0] - (h3[1] - h3[0])/2), abs(c[1] - (h3[1] - h3[0])/2)), reverse=False)[:20]

# constraint_res = []
# for i, j, _, p, d, s in constraints[:50]:
#     constraint_res.append(i + 1)
#     constraint_res.append(j + 1)
# constraint_res = np.unique(np.asarray(constraint_res))


best_res_constraint = {}
for ci, (i, j, prob, p, d, s) in enumerate(constraints):
    if not i in best_res_constraint:
        best_res_constraint[i] = constraints[ci]
    elif prob > best_res_constraint[i][2]:
        best_res_constraint[i] = constraints[ci]
constraints = best_res_constraint.values()

parser = PDBParser()
chains = list(list(parser.get_structure("target", target_pdb))[0].get_chains())
pdb_seq = []
res_seq = []
for chain in sorted(chains, key=lambda c: c.id):
    pdb_seq += [(chain.id, (str(residue.id[1]) + residue.id[2])) for residue in chain]

# for i in range(len(pdb_seq)):
#     print(i, seq[i], pdb_seq[i], res_seq[i])

# for n, (i, j, prob, p, d, s) in enumerate(constraints[:25]):
#     print("c{}:\t{} ({})\t{}".format(n + 1, p, s, d))

for n, (i, j, prob, p, d, s) in enumerate(constraints):
    res_i = "CA" if seq[i] == "G" else "CB"
    res_j = "CA" if seq[j] == "G" else "CB"
    print("distance {}_c{}, ({} and chain {} and name {} and resi {}), ({} and chain {} and name {} and resi {})".format(
        target_name, n + 1, target_name, pdb_seq[i][0], res_i, pdb_seq[i][1], target_name, pdb_seq[j][0], res_j, pdb_seq[j][1]
    ))
    print("select {}_r{}, ({} and chain {} and resi {} and name ca+cb)+({} and chain {} and resi {} and name ca+cb)".format(
        target_name, n + 1, target_name, pdb_seq[i][0], pdb_seq[i][1], target_name, pdb_seq[j][0], pdb_seq[j][1]
    ))
    if abs(p - d) < s:
        print("color {}, {}_c{}".format("green", target_name, n + 1))
    elif abs(p - d) < 2*s:
        print("color {}, {}_c{}".format("yellow", target_name, n + 1))
    else:
        print("color {}, {}_c{}".format("red", target_name, n + 1))

    print("set dash_width, {}, {}_c{}".format(int(prob * 15), target_name, n + 1))

print("group {}_constraint_lines, {}_c*".format(target_name, target_name))
print("group {}_constraint_atoms, {}_r*".format(target_name, target_name))
print("show sticks, {}_constraint_atoms".format(target_name))
print("select {}_loop_cdrh3, {} and chain H and resi {}-{} and name ca+cb".format(
    target_name, target_name, int(pdb_seq[h3[0]][1]), int(pdb_seq[h3[1]][1])
))
print("hide labels")