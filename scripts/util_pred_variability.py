from deeph3.util import get_probs_from_model, binned_matrix, get_dist_bins, protein_dist_matrix
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from deeph3.viz import plot_h3_range
import os


def build_mut_library(fasta_dir, mut_dir, fasta_name, parent_seq):
    with open(os.path.join(fasta_dir, fasta_name + ".fasta")) as parent_file:
        lines = parent_file.readlines()

    mut_files = [os.path.join(mut_dir, fasta_name + "_{}.fasta".format(i)) for i in range(len(parent_seq))]
    for i, mut_file in enumerate(mut_files):
        with open(mut_file, "w") as mut_file:
            for line in lines:
                if parent_seq in line:
                    mut_file.write(line.replace(parent_seq, parent_seq[:i] + "A" + parent_seq[i+1:]))
                else:
                    mut_file.write(line)

    return mut_files


def plot_mut_h3_loops(model, fasta_file, pdb_file):

    probs = get_probs_from_model(model, fasta_file, chain_delimiter=True)
    binned_mat = binned_matrix(probs, are_logits=False)

    # Convert to distances
    num_bins = binned_mat.shape[0]
    binvalues = get_dist_bins(num_bins)
    binvalues = [t[1] for t in binvalues]

    binned_mat_dist = binned_mat
    for i in range(0, num_bins):
        for j in range(0, num_bins):
            binned_mat_dist[i, j] = binvalues[binned_mat[i, j]]

    h3 = get_cdr_indices(pdb_file)['h3']
    plot_h3_range(binned_mat_dist[h3[0]:h3[1]+1,:], pdb_file, show_only_h3=True, color_min=4.0, color_max=16.0, out_file=fasta_file.replace('fasta', 'png'))


checkpoint_file = '/Users/jruffol1/Desktop/deep-H3-loop-prediction/deeph3/models/abantibody_train_antibody_validation_batchsize4model_1D3_2D10_bins26_AdaBound_lr0p0001_final_lr0p01_weight_decay0p0_CrossEntropyLoss_seed1234.p'
fasta_file =  '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/mut1.fasta'
pdb_file = '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut1.pdb'
fasta_dir = '/Users/jruffol1/Desktop/AMA_structures/fasta_dir'
mut_dir = '/Users/jruffol1/Desktop/AMA_structures/mut_dir'

# model = load_model(checkpoint_file)

parent = "SRWGGDGFYAMDY"
mut1 = "ARGRKYSSSFDY"
mut2 = "ARGGSFYYYYMDV"
mut3 = "AKLGIGYYYYGMDV"
mut4 = "ARGGAVAGTGVYYFDY"

# plot_mut_h3_loops(model, '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/parent.fasta', '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut2.pdb')
# plot_mut_h3_loops(model, '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/mut1.fasta', '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut1.pdb')
# plot_mut_h3_loops(model, '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/mut2.fasta', '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut2.pdb')
# plot_mut_h3_loops(model, '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/mut3.fasta', '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut3.pdb')
# plot_mut_h3_loops(model, '/Users/jruffol1/Desktop/AMA_structures/fasta_dir/mut4.fasta', '/Users/jruffol1/Desktop/AMA_structures/pdb_dir/mut4.pdb')


h3 = get_cdr_indices(pdb_file)['h3']
print(h3)
# dist_mat_N = protein_dist_matrix_N(pdb_file)[h3[0]:h3[1]+1,:]
# dist_mat_CA = protein_dist_matrix_CA(pdb_file)[h3[0]:h3[1]+1,:]
# dist_mat_CB = protein_dist_matrix(pdb_file)[h3[0]:h3[1]+1,:]
# dist_mat_C = protein_dist_matrix_C(pdb_file)[h3[0]:h3[1]+1,:]
# plot_h3_range(dist_mat_N, pdb_file, show_only_h3=True, color_min=4.0, color_max=16.0, out_file='/Users/jruffol1/Desktop/AMA_structures/png_dir/mut1_N.png')
# plot_h3_range(dist_mat_CA, pdb_file, show_only_h3=True, color_min=4.0, color_max=16.0, out_file='/Users/jruffol1/Desktop/AMA_structures/png_dir/mut1_CA.png')
# plot_h3_range(dist_mat_CB, pdb_file, show_only_h3=True, color_min=4.0, color_max=16.0, out_file='/Users/jruffol1/Desktop/AMA_structures/png_dir/mut1_CB.png')
# plot_h3_range(dist_mat_C, pdb_file, show_only_h3=True, color_min=4.0, color_max=16.0, out_file='/Users/jruffol1/Desktop/AMA_structures/png_dir/mut1_C.png')


# for mut_file in build_mut_library(fasta_dir, mut_dir, "parent", parent):
#     plot_mut_h3_loops(model, mut_file, pdb_file)