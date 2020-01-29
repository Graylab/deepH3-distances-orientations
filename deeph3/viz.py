import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D  # Not explicitly used, but needed for matplotlib 3d projections
from deeph3.util import binned_matrix, get_probs_from_model, get_dist_bins
from deeph3.preprocess.antibody_text_parser import get_cdr_indices
from pylatexenc.latexencode import utf8tolatex
import os

params = {
    'savefig.dpi': 150,
    'legend.fontsize': 'large',
    'axes.labelsize': 'large',
    'axes.titlesize':'large',
    'xtick.labelsize':'medium',
    'ytick.labelsize':'medium'
}
plt.rcParams.update(params)


def _get_outfilename(inputfile):
    extracted_filename = (inputfile.split('/')[-1]).split('.')[0]
    return extracted_filename

def plot_lines(y_vectors, x=None, labels=None, title='Loss', xlabel='',
               ylabel='', caption='', out_file=None):
    """Plots a line graph for each y vector in a list of y vectors"""
    # Ensure all y_vectors are the same length
    lens = np.array([len(_) for _ in y_vectors])
    if not np.all(len(y_vectors[0]) == lens):
        raise ValueError('All vectors in y_vectors must be the same length')

    # Default x and labels to array indices
    if x is None:
        x = range(len(y_vectors[0]))
    if labels is None:
        labels = [str(_) for _ in range(len(y_vectors[0]))]

    rc('text', usetex=False)
    full_xlabel = _add_caption(xlabel, caption)

    for y, label in zip(y_vectors, labels):
        plt.plot(x, y, label=utf8tolatex(label))

    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel(full_xlabel)
    plt.ylabel(ylabel)

    # Output to file or to screen
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    rc('text', usetex=False)

    plt.close()


def heatmap2d(matrix, title='Heatmap', ylabel='', xlabel='', caption='',
              color_min=None, color_max=None, out_file=None, line_indices=None,
              line_color='r',line_color_other='k',xticks=None,yticks=None,
              keep_open=False, cmap='viridis'):
    """Displays the heatmap of a 2D matrix

    :param matrix: The matrix to turn into a heat map.
    :type matrix: numpy.ndarray
    :param title: Title of the heat map.
    :type title: str
    :param caption: The caption under the x-axis
    :type caption: str
    :param color_min: The minimum value on the color bar.
    :param color_max: The maximum value on the color bar.
    :param out_file:
        The file to output to. If None, the heatmap is output to the screen.
    :param line_indices:
        A list of indices to add vertical and horizontal lines to.
    :param line_color: The color of the lines specified by line_indices.
    :param y_max: Max y value
    :param y_min: min y value
    :param x_max: Max x value
    :param x_min: Min x value
    :rtype: None
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if line_indices is None:
        line_indices = {}

    full_xlabel = _add_caption(xlabel, caption)

    #rc('text', usetex=False)
    plt.imshow(matrix, cmap=cmap)

    ax = plt.gca()

    if not xticks is None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        print(xticks)
    if not yticks is None:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)
        print(yticks)

    # Add color limits

    plt.colorbar()
    plt.clim(color_min, color_max)
    plt.title(title)
    plt.xlabel(full_xlabel)
    plt.ylabel(ylabel)

    # Explicitly set x and y limits (adding lines will extend the limits if this
    # is not done)
    plt.ylim(( len(matrix), 0))
    plt.xlim((0, len(matrix[0])))

    # Add horizontal and vertical lines
    for key in line_indices:
        list_indices=line_indices[key]
        if key=='h3':
            for idx in list_indices:
                plt.vlines(idx - 0.5, ymin=0, ymax=len(matrix[0]), color=line_color)
                plt.hlines(idx - 0.5, xmin=0, xmax=len(matrix), color=line_color)
        else:
            for idx in list_indices:
                plt.vlines(idx - 0.5, ymin=0, ymax=len(matrix[0]), color=line_color_other)
                plt.hlines(idx - 0.5, xmin=0, xmax=len(matrix), color=line_color_other)

    # Output to file or to screen
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    #rc('text', usetex=False)

    if not keep_open:
        plt.close()


def plot_output_grid(output, dist_range=None, angle_range=None):
    [out_dist, out_phi, out_phi, out_theta] = output

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    heatmap2d(binned_matrix(out_dist), title="dist", color_min=dist_range[0], color_max=dist_range[1], keep_open=True)
    fig.add_subplot(2, 2, 2)
    heatmap2d(binned_matrix(out_phi), title="phi", color_min=angle_range[0], color_max=angle_range[1], keep_open=True)
    fig.add_subplot(2, 2, 3)
    heatmap2d(binned_matrix(out_phi), title="phi", color_min=angle_range[0], color_max=angle_range[1], keep_open=True)
    fig.add_subplot(2, 2, 4)
    heatmap2d(binned_matrix(out_theta), title="theta", color_min=angle_range[0], color_max=angle_range[1], keep_open=True)
    plt.tight_layout()
    plt.close()

    return fig


def plot_probs_list(probs_list, hide_indices=None, vlines=None, vline_color='r',
                    title='Probability Distributions', xlabel='x',
                    ylabel='y', zlabel='P(x)', xticks=None, yticks=None,
                    xtick_step=1, ytick_step=1, shadow=True,
                    show_xtick_labels=True, show_ytick_labels=True,
                    show_curve=True, show_bars=False, curve_color='tab:orange',
                    bar_color=None, bar_alpha=0.4, outfile=None):
    """
    :param probs_list: The list of probability distributions to plot.
    :type probs_list: List[List[float]]
    :param vlines: A list of x, y coordinates to draw vertical lines under the
                   z curve. Should be shape [n, 2]
    :type vlines: List[List[int]]
    :param vline_color: Color of the vertical lines.
    :param title: Title of the plot.
    :type title: str
    :param xlabel: Label of the x axis.
    :type xlabel: str
    :param ylabel: Label of the y axis.
    :type ylabel: str
    :param zlabel: Label of the z axis.
    :type zlabel: str
    :param xticks: Tick labels to put on the x-axis.
    :type xticks: List[str]
    :param yticks: Tick labels to put on the y-axis.
    :type yticks: List[str]
    :param ytick_step: The step size of the ticks on the y-axis.
    :type ytick_step: int
    :param xtick_step: The step size of the ticks on the x-axis.
    :type xtick_step: int
    :param show_xtick_labels: Whether or not to show the x tick labels.
    :type show_xtick_labels: bool
    :param show_ytick_labels: Whether or not to show the y tick labels.
    :type show_ytick_labels: bool
    :param show_curve: Whether or not to show the curve of the probabilities.
    :type show_curve: bool
    :param show_bars: Whether or not to show the bar graph of the probabilities.
    :type show_bars: bool
    :param curve_color: Color of the curves.
    :type curve_color: str
    :param bar_color: Color of the bars in the bar plot.
    :type bar_color: str
    :param bar_alpha: Transparency of bars (0.0 transparent through 1.0 opaque)
    :rtype: None
    """
    hide_indices = hide_indices if hide_indices is not None else []

    rc('text', usetex=False)

    preamble = 'text.latex.preamble'
    preamble_packages = [r'\usepackage{amsmath}', r'\usepackage{siunitx}']
    if len(mpl.rcParams[preamble]) == 0:
        mpl.rcParams[preamble] = preamble_packages
    else:
        mpl.rcParams[preamble].append(preamble_packages)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in range(len(probs_list)):
        if x in hide_indices:
            # Plot a transparent point so the tick still shows in the figure
            ax.plot([x], [0], zs=[0], alpha=0.0)
            continue
        ys = list(range(len(probs_list[x])))
        xs = [x for _ in range(len(probs_list[x]))]
        zs = probs_list[x]
        if show_bars:
            if bar_color is None:
                ax.bar(xs, ys, zs=zs, color=bar_color, alpha=bar_alpha)
            else:
                ax.bar(xs, ys, zs=zs, alpha=bar_alpha)
        if show_curve:
            ax.plot(xs, ys, zs=zs,  color=curve_color)
        if shadow:
            ax.plot(xs, ys, zs=0,  color='gray', alpha=0.5)

    # Add vertical lines at specified indices
    if vlines is not None:
        for x, y in vlines:
            if x in hide_indices:
                continue
            zs = [0, probs_list[x][y]]
            xs = [x, x]
            ys = [y, y]
            ax.plot(xs, ys, zs=zs, color=vline_color)

    ax.set_title(utf8tolatex(title))
    ax.set_xlabel(utf8tolatex(xlabel))
    ax.set_ylabel(ylabel)
    ax.set_zlabel(utf8tolatex(zlabel))

    # Plot x ticks
    x = list(range(len(probs_list)))
    xticks = xticks if xticks is not None else x
    xticks = xticks if show_xtick_labels else []
    ax.set_xticks(xticks)
    ax.set_xticks(x[::xtick_step])
    ax.get_xaxis().set_ticklabels(xticks[::xtick_step])

    # Plot y ticks
    y = list(range(len(probs_list[0])))
    yticks = yticks if yticks is not None else y
    yticks = yticks if show_ytick_labels else []
    ax.set_yticks(y[::ytick_step])
    ax.get_yaxis().set_ticklabels(yticks[::ytick_step],
                                  fontsize='x-small')

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
    plt.close()

    rc('text', usetex=False)


def plot_probs(probs, title='Probability Distribution', ylabel='P(x)',
               xlabel='x', caption='', ticks=None, show_xtick_labels=True,
               show_curve=True, show_bars=True, curve_color='tab:orange',
               bar_color='tab:blue'):
    if ticks is None:
        ticks = list(range(len(probs)))
    else:
        ticks = [utf8tolatex(_) for _ in ticks]

    rc('text', usetex=False)

    fig, ax = plt.subplots()
    x = list(range(len(probs)))
    if show_curve:
        ax.plot(x, probs, color=curve_color)
    if show_bars:
        ax.bar(ticks, probs, color=bar_color)

    ax.set_title(utf8tolatex(title))
    ax.set_ylabel(utf8tolatex(ylabel))
    ax.set_xlabel(_add_caption(xlabel, caption))

    # Set tick labels
    ticks = ticks if show_xtick_labels is not None else []
    ax.set_xticks(x)
    ax.get_xaxis().set_ticklabels(ticks)

    fig.autofmt_xdate()

    plt.show()
    rc('text', usetex=False)

    plt.close()


def plot_binned_distance_matrix(model, fasta_file, **kwargs):
    """Plots the distance matrix produces by a model.
    :param model: The model to produce the distance matrix output.
    :param fasta_file: The fasta file to extract input from
    :param kwargs: Keyword arguments to feed into heatmap2d.
    :return:
    """
    state_dict = model.state_dict()
    in_planes = state_dict[list(state_dict.keys())[0]].shape[1]
    chain_delimiter = (in_planes == 21)

    probs = get_probs_from_model(model, fasta_file, chain_delimiter=chain_delimiter)
    binned_dists = binned_matrix(probs, are_logits=False)
    heatmap2d(binned_dists, **kwargs)


def plot_losses(checkpoint_file, title='Loss', epoch_only=False, **kwargs):
    c = torch.load(checkpoint_file, map_location='cpu')
    # Get data to plot
    val = np.array(c['validation_losses'])[:, 2]
    train = np.array(c['training_losses'])[:, 2]

    # Get x-axis tick labels
    batches = np.array(c['training_losses'])[:, 1]
    epochs = np.array(c['training_losses'])[:, 0]
    x = ['{}.{}'.format(int(e), int(b)) for e, b in zip(epochs, batches)]

    # Remove batch data if epoch only
    if epoch_only:
        epochs_seen = set()
        epoch_ticks = []
        new_val = []
        new_train = []
        for i, label in enumerate(x):
            epoch = label.split('.')[0]
            if epoch not in epochs_seen:
                epochs_seen.add(epoch)
                epoch_ticks.append(epoch)
                new_val.append(val[i])
                new_train.append(train[i])
        x = epoch_ticks
        val = new_val
        train = new_train

    plot_lines([train, val], x=x, labels=['training', 'validation'],
               xlabel='Epochs', ylabel='Loss', title=title, **kwargs)


def plot_metrics(checkpoint_file, title='Metrics', **kwargs):
    """
    :param checkpoint_file:
        A pytorch serialized model (output of torch.save()) that contains
        these keys: 'validation_metrics', 'training_metrics',
        'training_losses', 'validation_losses.

        Here, 'validation_metrics' and 'training' metrics contain a list of
        dictionaries for each saved batch/epoch where the key is the metric
        name and the value is the metrics value for that particular save point.
        For example, for a model with two save points tracking precision
        and recall:
        >>> cp = torch.load(checkpoint_file)
        >>> cp['training_metrics']
        [{'precision': 0.9, 'recall': 0.4}, {'precision': 0.93, 'recall': 0.5}]
        >>> cp['validation_metrics']
        [{'precision': 0.7, 'recall': 0.2}, {'precision': 0.8, 'recall': 0.3}]

    :param title:
    :param kwargs:
    :return:
    """
    c = torch.load(checkpoint_file, map_location='cpu')
    # Get data to plot
    val_dicts = np.array(c['validation_metrics'])
    train_dicts = np.array(c['training_metrics'])
    all_val_metrics = {k: [] for k in val_dicts[0]}
    all_train_metrics = {k: [] for k in train_dicts[0]}
    for vd, td in zip(val_dicts, train_dicts):
        for k in vd.keys():
            all_val_metrics[k].append(vd[k])
        for k in td.keys():
            all_train_metrics[k].append(td[k])

    # Get all labels and lines to plot
    train_metric_labels = [str(k) for k in all_train_metrics.keys()]
    train_metrics = [all_train_metrics[k] for k in all_train_metrics.keys()]
    val_metric_labels = [str(k) for k in all_val_metrics.keys()]
    val_metrics = [all_val_metrics[k] for k in all_val_metrics.keys()]

    # Get x-axis tick labels
    batches = np.array(c['training_losses'])[:, 1]
    epochs = np.array(c['training_losses'])[:, 0]
    x = ['{}.{}'.format(int(e), int(b)) for e, b in zip(epochs, batches)]

    print(train_metric_labels)
    print(train_metrics)

    plot_lines(train_metrics, x=x, xlabel='Epochs', labels=train_metric_labels,
               ylabel='Metric Value', title='Training '+title, **kwargs)
    plot_lines(val_metrics, x=x, xlabel='Epochs', labels=val_metric_labels,
               ylabel='Metric Value', title='Validation '+title, **kwargs)


def plot_h3_range(dist_mat, ab_pdb_path, show_only_h3=False, **kwargs):
    """
    :param dist_mat: The L x L distance matrix of the antibody.
    :type dist_mat: torch.Tensor
    :param ab_pdb_path: The chothia numbered antibody file.
    :param show_only_h3:
        Whether to show only the h3 loop residues. If False, the entire distance
        matrix is displayed with red lines around the h3 residues, essentially
        highlighting them.
    :param kwargs:
    :return:
    """
    indices = get_cdr_indices(ab_pdb_path)
    h3 = indices['h3']
    l3 = indices['l3']

    if show_only_h3:
        heatmap2d(dist_mat[:, h3[0]:h3[1]+1], line_indices=indices,**kwargs)
    else:
        heatmap2d(dist_mat, line_indices=indices, **kwargs)


def plot_h3_region(dist_mat, ab_pdb_path, **kwargs):
    """
    :param dist_mat: The L x L distance matrix of the antibody.
    :type dist_mat: torch.Tensor
    :param ab_pdb_path: The chothia numbered antibody file.
    :param kwargs:
    :return:
    """
    indices = get_cdr_indices(ab_pdb_path)
    h3 = indices['h3']

    heatmap2d(dist_mat[h3[0]:h3[1]+1, h3[0]:h3[1]+1], line_indices=indices,**kwargs)


def plot_cross_loops(dist_mat, ab_pdb_path,**kwargs):
    """
    :param dist_mat: The n x n distance matrix of the antibody.
    :type dist_mat: torch.Tensor
    :param ab_pdb_path: The chothia numbered antibody file.
    :param kwargs:
    :return:
    """
    indices = get_cdr_indices(ab_pdb_path)
    h3 = indices['h3']
    xticks = [str(t) for t in range(h3[0] , h3[1]+1)]

    for key in indices:
        curloop = indices[key]
        filename_base = _get_outfilename(ab_pdb_path)
        outfile = 'results/heatmaps/heatmap2DCrossLoop_%s_%s.png' %(key,filename_base)
        print (outfile)

        yticks=[str(t) for t in range(curloop[0],curloop[1]+1)]

        heatmap2d(dist_mat[ curloop[0] : curloop[1]+1, h3[0] : h3[1]+1], line_indices=indices,title = '%s - h3' %key, \
                  xticks= xticks, yticks=yticks, out_file = outfile,**kwargs)

def plot_residue_prob_dists(residue_index, prob_matrix, true_binned_dist_matrix,
                            start=0, end=None, xlabel='Residue',
                            ylabel=r'Distance Range (\si{\angstrom})',
                            zlabel='P(Distance)',
                            ytick_step=6, curve_color='tab:blue',
                            hide_indices=None, **kwargs):
    """
    :param residue_index: The residue to plot the probability distributions of.
    :param prob_matrix: The LxL matrix of pairwise residue
    :param true_binned_dist_matrix:
    :param start:
    :param end:
    :param xlabel: Label of the x axis.
    :type xlabel: str
    :param ylabel: Label of the y axis.
    :type ylabel: str
    :param zlabel: Label of the z axis.
    :type zlabel: str
    :param ytick_step:
    :param curve_color:
    :param hide_indices:
    :param kwargs:
    :return:
    """
    end = end if end is not None else len(prob_matrix)

    # Get the probability distribution of the target residue
    residue_probs = prob_matrix[residue_index].detach()
    residue_probs = residue_probs[start:end]

    # Get labels for the yticks
    num_bins = len(residue_probs[0])
    bins = [str(_) for _ in get_dist_bins(num_bins)]

    # For each residue i, get its true binned distance to residue_index
    true_bins = true_binned_dist_matrix[start:end, residue_index]
    true_bins = [[i, v] for i, v in enumerate(true_bins)]

    # Number residues according to start and end points
    x = list(range(start, end))

    # Renumber indices to hide
    if hide_indices is not None:
        hide_indices = [i - start for i in hide_indices if i - start >= 0]

    plot_probs_list(residue_probs, vlines=true_bins, curve_color=curve_color,
                    xticks=x, yticks=bins, ytick_step=ytick_step,
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    hide_indices=hide_indices, **kwargs)


def _add_caption(label, caption):
    """Adds a caption to a label using LaTeX"""
    full_label = ''
    if label != '' or caption != '':
        if label != '':
            label = utf8tolatex(label)
            label = label + r'\\*'
        if caption != '':
            caption = utf8tolatex(caption)
            caption = r'\textit{\small{' + caption + r'}}'
        full_label = r'\begin{center}' + label + caption + r'\end{center}'
    return full_label


if __name__ == '__main__':
    def main():
        '''
        from deeph3.util import get_bins

        probs = np.array([0, 0.10, 0.50, 0.25, 0.10, 0.05, 0])
        vlines = np.array([[0, 4], [1, 2], [3, 6], [4, 0], [5, 3]])

        yticks = [str(_) for _ in get_bins(len(probs))]
        probs_list = [probs]
        np.random.seed(0)
        for _ in range(5):
            np.random.shuffle(probs)
            probs_list.append(probs.copy())

        plot_probs_list(probs_list, show_bars=False, vlines=vlines,
                        yticks=yticks)
        '''
        import torch
        from deeph3 import load_model
        from deeph3.util import get_probs_from_model, binned_matrix, get_dist_bins
        from deeph3.preprocess.antibody_text_parser import get_cdr_indices
        checkpoint_file = "deeph3/models/adam_opt_lr01_dil2/abantibody_train_antibody_validation_batchsize4model_1D3_2D25_dil5_bins26_Adam_lr0p001_weight_decay0p0_CrossEntropyLoss_ReduceLROnPlateau_factor0p01_min_lr0p0_patience1_threshold0p01_seed1234.p"
        fasta_file =  "/home/jruffol1/kic_working/mut1/mut1.fasta"
        pdb_file = "/home/jruffol1/kic_working/mut1/mut1.pdb"
        out_file = "/home/jruffol1/kic_working/mut1probs.png"

        model = load_model(checkpoint_file)
        prob_mats = get_probs_from_model(model, fasta_file, chain_delimiter=True)
        prob_mats[0] = (prob_mats[0] + prob_mats[0].transpose(0, 1)) / 2

        prob_mat_dist = prob_mats[0]
        binned_mat_dist = binned_matrix(prob_mats, are_logits=False)[0]

        indices = get_cdr_indices(pdb_file)
        h3 = indices['h3']

        #heatmap2d(binned_mat, color_min=0, color_max=26, title='Predicted ', line_indices=[h3[0], h3[1]+1])
        #heatmap2d(binned_mat, color_min=0, color_max=26, title='Predicted '+basename(fasta_file), line_indices=[(h3[0] + h3[1]) // 2, ((h3[0] + h3[1]) // 2)+1])

        #plot_cross_loops(binned_mat_dist,pdb_file)
        start = h3[0]
        end = h3[1]
        #start = 0
        #end = None
        idx = (h3[0] + h3[1]) // 2
        hide_indices = [i for i, _ in enumerate(prob_mat_dist[idx]) if max(_) > 0.99 or max(_) < 0.13]

        plot_residue_prob_dists(idx, prob_mat_dist, binned_mat_dist, title="Predicted Distance from Residue {}".format(idx), 
                                start=start, end=end, hide_indices=hide_indices, xtick_step=2,
                                ylabel="Distance Range (A)", outfile=out_file)
    main()

