import os
import sys
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.cluster.vq import kmeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



# params = {
#     'savefig.dpi': 200,
#     'legend.fontsize': '24',
#     'axes.labelsize': '20',
#     'axes.titlesize':'20',
#     'xtick.labelsize':'18',
#     'ytick.labelsize':'18'
# }
# plt.rcParams.update(params)

experiment_dir = sys.argv[1]

plot_files = ["energy_funnel.pdf", "funnel.pdf", "cst_funnel.pdf", "corr_funnel.pdf"]
plot_xlabel = "RMSD (Å)"
# plot_ylabels = ["total_score - atom_pair_constraint", "total_score", "atom_pair_constraint", "rosetta_energy + C * atom_pair_constraint"]
plot_ylabels = ["REU", "REU", "REU", "REU"]
plot_titles = ["Rosetta Energy", "Total Energy", "Constraint Energy", "Correlation-Adjusted Total Energy"]
plot_abbr = ["RE", "TE", "CE", "CATE"]

topns = [1, 5]


directories = sorted(list(glob(os.path.join(experiment_dir, "*"))))
directories = [d for d in directories if os.path.isdir(d) and os.path.exists(os.path.join(d, "score.sc"))]
stats = np.empty(shape=(len(directories), 6 + (len(topns) + 1) * len(plot_titles)), dtype="U20")
for directory_i, directory in tqdm(enumerate(directories)):
    score_file = os.path.join(directory, "score.sc")
    if not os.path.isdir(directory) or not os.path.exists(score_file):
        stats[directory_i] = '-1'
        continue

    constraint_num = 1
    with open(os.path.join(directory, "constraints")) as constraint_file:
        constraint_num = len(constraint_file.readlines())

    target_name = os.path.split(glob(os.path.join(directory, "*.fasta"))[0])[1][:-6]
    # target_name = os.path.split([t for t in glob(os.path.join(directory, "*.pdb")) if "trunc" in t][0])[1][:4]
    
    bests_dir = os.path.join(directory, "top_pdbs")
    if os.path.exists(bests_dir):
        os.system("rm -rf {}".format(bests_dir))
    os.mkdir(bests_dir)
    
    fig_dir = os.path.join(directory, "figures")
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    with open(score_file) as file:
        cols = file.readlines()[1].split()
    scores = np.loadtxt(score_file, skiprows=2, usecols=[
                        cols.index("bb_heavy_rmsd"), 
                        cols.index("total_score"), 
                        cols.index("atom_pair_constraint"),
                        cols.index("angle_constraint"),
                        cols.index("dihedral_constraint")]).T
    cscores = scores[2] + scores[3] + scores[4]
    scores[2] = cscores
    scores[2] /= constraint_num
    descs = np.loadtxt(score_file, skiprows=2, usecols=[cols.index("description")], dtype=str).T

    has_native = os.path.exists(os.path.join(directory, "native_score.sc"))
    if has_native:
        native_score_file = os.path.join(directory, "native_score.sc")
        with open(native_score_file) as file:
            cols = file.readlines()[1].split()
        native_scores = np.loadtxt(native_score_file, skiprows=2, usecols=[
                                   cols.index("bb_heavy_rmsd"), 
                                   cols.index("total_score"), 
                                   cols.index("atom_pair_constraint"),
                                   cols.index("angle_constraint"),
                                   cols.index("dihedral_constraint")]).T
        ncscores = native_scores[2] + native_scores[3] + native_scores[4]
        native_scores[2] = ncscores
        native_scores[2] /= constraint_num

    xdata = scores[0]

    re_ce_scores = np.asarray([scores[1] - cscores, scores[2]])
    
    # re_ce_scores = re_ce_scores[:, np.flip(re_ce_scores[2].argsort())]
    # re_ce_scores = re_ce_scores[:, :len(re_ce_scores[0]) // 4]

    # re_ce_scale = np.max(re_ce_scores, axis=1)
    # re_ce_scores[0] /= re_ce_scale[0]
    # re_ce_scores[1] /= re_ce_scale[1]
    
    # sil = []
    # for k in range(2, 5):
    #     kmean = KMeans(n_clusters = k).fit(re_ce_scores[:2].T)
    #     labels = kmean.labels_
    #     sil.append(silhouette_score(re_ce_scores[:2].T, labels, metric = 'euclidean'))

    # bestk = np.argmax(sil) + 2
    # if bestk > 2:
    #     # clusters = [np.mean(re_ce_scores[:2,:len(re_ce_scores[0])//4], axis=1), np.mean(re_ce_scores[:2,:-1 * len(re_ce_scores[0])//4], axis=1)]
    #     clusters = kmeans(re_ce_scores[:2].T, bestk, iter=20)[0]
    #     re_ce_scores = np.asarray([r for r in re_ce_scores.T if np.linalg.norm(r[:2] - clusters[0]) < np.linalg.norm(r[:2] - clusters[1])]).T
    # re_ce_scores[0] *= re_ce_scale[0]
    # re_ce_scores[1] *= re_ce_scale[1]

    # re_ce_scores = re_ce_scores[:, np.logical_and(re_ce_scores[0] < np.quantile(re_ce_scores[0], 0.5), re_ce_scores[1] < np.quantile(re_ce_scores[1], 0.5))]
    # re_ce_scores = re_ce_scores[:, re_ce_scores[1] < np.median(re_ce_scores[1])]
    # re_ce_scores = re_ce_scores[:, re_ce_scores[0] < np.median(re_ce_scores[0])]

    corr = np.corrcoef(re_ce_scores)[1,0]
    cst_scale = (-1 / np.log(corr)) if corr > 0 else 0

    ydata = [scores[1] - cscores, scores[1], scores[2], scores[1] - cscores + cst_scale * scores[2]]
    
    xmin = np.min(re_ce_scores[1])
    ymin = np.min(re_ce_scores[0])
    xwidth = abs(xmin - np.max(re_ce_scores[1]))
    ywidth = abs(ymin - np.max(re_ce_scores[0]))
    ax = plt.subplot(111)
    ax.scatter(ydata[2], ydata[0], s=5)
    # ax.scatter(re_ce_scores[1], re_ce_scores[0], s=5)
    
    # ax.scatter(clusters.T[1], clusters.T[0], s=10)
    # ax.add_patch(patches.Rectangle((xmin, ymin), xwidth, ywidth,fill=False, color='r', linewidth=3)) 
    plt.xlabel("CE REU")
    plt.ylabel("RE REU")
    plt.ylim([min(scores[1] - cscores), max(scores[1] - cscores)])
    plt.xlim([min(scores[2]), max(scores[2])])
    plt.title("Rosetta Energy vs Constraint Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "RE_CE_plot.pdf"))
    plt.close()


    if has_native:
        nxdata = native_scores[0]
        nydata = [native_scores[1] - ncscores, native_scores[1], native_scores[2], native_scores[1] - ncscores + cst_scale * native_scores[2]]

    print("\n", '*' * 30, directory, "-", target_name, '*' * 30, "\n")
    print("Total: {}".format(len(descs)))
    best_rmsd = xdata[np.argmin(xdata)]
    print("Best rmsd sampled: {} ({}, {})".format(os.path.split(descs[np.argmin(
        xdata)])[1], best_rmsd, ydata[0][np.argmin(xdata)]))
    print("Correlation: {}".format(round(corr, 3)))
    print("NatCorr: {}".format(round(cst_scale, 3)))

    os.system("cp {}.pdb {}".format(descs[np.argmin(xdata)], bests_dir))
    os.system("cp {}.pdb {}".format(os.path.join(directory, target_name), bests_dir))
    
    sub_angstrom = len(np.where(xdata < 1)[0])
    f_sub_angstrom = round(sub_angstrom / len(xdata), 3)
    print("SubA sampled: {} ({}/{})".format(f_sub_angstrom, sub_angstrom, len(xdata)))
    sub_2angstrom = len(np.where(xdata < 2)[0])
    f_sub_2angstrom = round(sub_2angstrom / len(xdata), 3)
    print("Sub2A sampled: {} ({}/{})".format(f_sub_2angstrom, sub_2angstrom, len(xdata)))

    for i in range(len(plot_titles)):
        print("\n{}".format(plot_titles[i]))
        print("\nCorr: {}".format(np.corrcoef([xdata, ydata[i]])[1,0]))
        funnel_file = os.path.join(fig_dir, plot_files[i])

        os.mkdir(os.path.join(bests_dir, plot_abbr[i]))

        stats[directory_i][0] = os.path.split(directory)[1]
        stats[directory_i][1] = target_name
        stats[directory_i][2] = len(xdata)
        stats[directory_i][3] = str(f_sub_angstrom)
        stats[directory_i][4] = str(f_sub_2angstrom)
        stats[directory_i][5] = str(best_rmsd)

        x_range = [0, max(xdata) * 1.2]
        y_range = [min(ydata[i]), max(ydata[i])]

        plt.figure(figsize=(4, 4))
        plt.scatter(xdata, ydata[i], s=5)

        if has_native:
            x_range = [0, max(x_range[1], max(nxdata))]
            y_range = [min(y_range[0], min(nydata[i])), max(y_range[1], max(nydata[i]))]

            plt.scatter(nxdata, nydata[i], s=15)

        topn_list = []
        for j, topn in enumerate(topns):
            if not len(ydata[i]) > topn:
                continue
            topn_indices = np.unique(
                np.hstack([np.where(ydata[i] == np.sort(ydata[i])[n]) for n in range(topn)]))
            topn_values = np.asarray([xdata[n] for n in topn_indices])

            if topn == 5:
                topn_ys = np.asarray([ydata[i][n] for n in topn_indices])
                plt.scatter(topn_values, topn_ys, s=15, c='r')
                load_commands = "load {}.pdb\ncolor gray90, {}\nalign {}, mut1\nload {}.pdb\ncolor slate, {}\n\nalign {}, mut1\n".format(
                    target_name,
                    target_name,
                    target_name,
                    os.path.split(descs[np.argmin(xdata)])[1],
                    os.path.split(descs[np.argmin(xdata)])[1],
                    os.path.split(descs[np.argmin(xdata)])[1]
                )

                for topn_i in topn_indices:
                    os.system("cp {}.pdb {}".format(descs[topn_i], os.path.join(bests_dir, plot_abbr[i])))

                    load_commands += ("load {}/{}.pdb\ngroup {}_top5, {}\nalign {}, mut1\n".format(
                        plot_abbr[i],
                        os.path.split(descs[topn_i])[1], 
                        target_name,
                        os.path.split(descs[topn_i])[1], 
                        os.path.split(descs[topn_i])[1], 
                        target_name
                    ))

                load_commands += "color salmon, {}_top5".format(target_name)
                with open(os.path.join(bests_dir, plot_abbr[i], "load_script"), "w") as load_script:
                    load_script.write(load_commands)

            topn_list.append(np.min(topn_values))
            print("Top{} rmsd by {}: {} ({})".format(topn, plot_titles[i], os.path.split(
                descs[topn_indices[np.argmin(topn_values)]])[1], np.min(topn_values)))
            stats[directory_i][6 + (len(topns) + 1) * i + j + 1] = str(np.min(topn_values))
    
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabels[i])
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.title(plot_titles[i])
        plt.tight_layout()
        plt.savefig(funnel_file)
        plt.close()

np.savetxt(os.path.join(experiment_dir, "stats.csv"), stats, fmt="%s", delimiter=",")

energy_comparison_dir = os.path.join(experiment_dir, "energy_comparison")
# energy_comparison_dir = os.path.join(experiment_dir, "energy_comparison_labeled")
if not os.path.exists(energy_comparison_dir):
    os.mkdir(energy_comparison_dir)

horizontal = False
for i, title in enumerate(plot_titles):
    if i == 0:
        continue
    
    if horizontal:
        plt.figure(figsize=(4 * len(topns), 5))
    else:
        plt.figure(figsize=(3.5, 4 * len(topns)))
    
    for j, topn in enumerate(topns):
        if horizontal:
            plt.subplot(1, len(topns), j + 1)
            plt.xlabel("{} RMSD (Å)".format(plot_abbr[0]))
            if j == 0:
                plt.ylabel("{} RMSD (Å)".format(plot_abbr[i]))
        else:
            plt.subplot(len(topns), 1, j + 1)
            plt.xlabel("{} RMSD (Å)".format(plot_abbr[0]))
            plt.ylabel("{} RMSD (Å)".format(plot_abbr[i]))

        plt.scatter(stats.T[6 + j + 1].astype(np.float), stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float))
        # for l, x, y in zip(stats.T[1], stats.T[6 + j + 1].astype(np.float), stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float)):
        #     plt.annotate(l[:4], (x, y))
        plt.plot((0, 6), (0, 6), 'k-')

        plt.title("Top {}".format(topn))
        
        plt.xlim((0, 6))
        plt.ylim((0, 6))

    # plt.title("{} vs {}".format(title, plot_titles[0]))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(energy_comparison_dir, "{}_figure.pdf".format(plot_abbr[i])))
    plt.close()

plt.figure(figsize=(10, 10), dpi=200)
for i, title in enumerate(plot_titles):
    if i == 0:
        continue

    for j, topn in enumerate(topns):
        plt.subplot(len(topns), len(plot_titles) - 1, (i - 1) * (len(topns)) + j + 1)
        plt.scatter(stats.T[6 + j + 1].astype(np.float), stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float))
        # for l, x, y in zip(stats.T[1], stats.T[6 + j + 1].astype(np.float), stats.T[6 + len(plot_titles) * i + j + 1].astype(np.float)):
        #     plt.annotate(l, (x, y))
        plt.plot((0, 6), (0, 6), 'k-')

        if i == 1:
            plt.title("Top {}".format(topn))
        if i == len(plot_titles) - 1:
            plt.xlabel("{} RMSD (Å)".format(plot_abbr[0]))
        if j == 0:
            plt.ylabel("{} RMSD (Å)".format(plot_abbr[i]))
        plt.xlim((0, 6))
        plt.ylim((0, 6))

        print("{} {}: \n\tWorse:\t{}/{}\n\tSame:\t{}/{}\n\tBetter:\t{}/{}".format(title, topn, 
            sum(stats.T[6 + j + 1].astype(np.float) < stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float)), len(stats.T[6 + j + 1].astype(np.float)),
            sum(stats.T[6 + j + 1].astype(np.float) == stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float)), len(stats.T[6 + j + 1].astype(np.float)),
            sum(stats.T[6 + j + 1].astype(np.float) > stats.T[6 + (len(topns) + 1) * i + j + 1].astype(np.float)), len(stats.T[6 + j + 1].astype(np.float))))

plt.tight_layout()
plt.savefig(os.path.join(energy_comparison_dir, "all_figure.pdf"))