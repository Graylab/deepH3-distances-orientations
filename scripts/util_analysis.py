import torch
from deeph3 import load_model
from deeph3.util import get_probs_from_model, binned_matrix, get_dist_bins
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from deeph3.viz import plot_cross_loops

def test_cross_loops_with_args( model , fasta_file, pdb_file, plot_matrices=True):

    probs = get_probs_from_model(model, fasta_file, chain_delimiter=True)
    binned_mat = binned_matrix(probs, are_logits=False)[0]

    # Convert to distances
    num_bins = binned_mat.shape[0]
    binvalues = get_dist_bins(num_bins)
    binvalues = [t[1] for t in binvalues]

    binned_mat_dist = binned_mat
    for i in range(0, num_bins):
        for j in range(0, num_bins):
            binned_mat_dist[i, j] = binvalues[binned_mat[i, j]]

    indices = get_cdr_indices(pdb_file)
    h3 = indices['h3']

    if plot_matrices:
        plot_cross_loops(binned_mat_dist, pdb_file, color_min=4.0, color_max=16.0)



if __name__ == '__main__':
    def main():

        checkpoint_file = '/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/deep-H3-loop-prediction/deeph3/models/adam_opt_lr01/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p'
        fasta_file =  "/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/kic_working/mut1/mut1.fasta"
        pdb_file = "/home-2/jruffol1@jhu.edu/work/jruffol1@jhu.edu/kic_working/mut1/mut1.pdb"

        model = load_model(checkpoint_file)
        test_cross_loops_with_args(model,fasta_file,pdb_file)
    main()