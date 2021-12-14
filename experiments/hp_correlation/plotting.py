import matplotlib.pyplot as plt
import copy
import sys
from precomputation import results_order, load_correlation_dict, dd_list
import os
import pickle


def save_fig(correlation_dict, name, plot_measures, error_type, min_runs=20):
    plt.figure(figsize=(20, 15), dpi=300)
    for k, epochs in correlation_dict.items():
        if k.lower()[11:] not in plot_measures:
            continue
        epochs = copy.deepcopy(epochs)
        epochs.pop("final")
        l = sorted(epochs.items())
        x, corrs = zip(*l)
        indices = [i for i, j in enumerate(corrs) if len(j) >= min_runs]
        x = [x[i] for i in indices]
        corrs = [corrs[i] for i in indices]
        y = []
        if error_type == "avg":
            y = [sum(c)/len(c) for c in corrs]
        elif error_type == "min":
            y = [min(c, key=abs) for c in corrs]
        plt.plot(x, y, label=(k[11:] if k.lower()[:10] == "complexity" else k))
    plt.legend(bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig('figures/{}_{}_best'.format(name, min_runs))


def box_plot(measure: str, epoch_correlations_dict, min_corrs_epoch: int):
    corrs_list = []
    labels = []

    for epoch, corrs in sorted(epoch_correlations_dict.items()):
        if len(corrs) < min_corrs_epoch:
            continue

        corrs_list.append(corrs)
        epoch = "final" if epoch == sys.maxsize else epoch
        labels.append(epoch)

    fig, ax = plt.subplots()
    ax.set_title(measure)
    ax.boxplot(corrs_list, labels=labels)

    return ax


def get_figs_dir(hp: str, corr_type: str, min_corrs_epoch: int = 0, fill_nan_epochs: bool = True,
                 filter_train_acc: float = 0):
    parent_path = './results/{}/min_corrs_{}-fill_nan_epochs_{}-min_acc_{}-figs'.format(
        hp, min_corrs_epoch, fill_nan_epochs, filter_train_acc)
    dir_path = '{}/{}'.format(parent_path, corr_type)

    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
        os.mkdir(dir_path)
    elif not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


def correlations_box_plot(hp: str, corr_type: str, min_corrs_epoch: int = 0, fill_nan_epochs: bool = True,
                          filter_train_acc: float = 0):
    correlation_dict = load_correlation_dict(hp, corr_type, fill_nan_epochs, filter_train_acc)

    plot_order = results_order[:]
    plot_order.append(hp)

    min_corrs_epoch = len(correlation_dict[hp][1]) if min_corrs_epoch == 0 else min_corrs_epoch

    dir_path = get_figs_dir(hp, corr_type, min_corrs_epoch, fill_nan_epochs, filter_train_acc)

    for k in plot_order:
        measure_epoch_corrs = correlation_dict[k]
        ax = box_plot(k, measure_epoch_corrs, min_corrs_epoch)
        ax.figure.savefig('{}/{}.pdf'.format(dir_path, k.replace('/', '_')))
        plt.close()


correlations_box_plot("batch_size", "acc", 0, True, 0.99)
