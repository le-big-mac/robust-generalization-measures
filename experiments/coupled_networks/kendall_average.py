import wandb
import pandas as pd
import scipy.stats as stats
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import copy

api = wandb.Api()
runs = api.runs("rgm")
hparams = ['lr', 'seed', 'data_seed', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob', 'weight_decay']
config_params = ['lr', 'batch_size', 'model_depth', 'seed']
config_values = {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1], 'batch_size': [32, 64, 128, 256],
                 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43]}


def get_kendall(hp: str, plot_measures: list):
    fixed_params = hparams
    fixed_params.remove(hp)

    run_dict = defaultdict(list)
    for r in runs:
        if r.state == "running" or r.config["batch_norm"] or r.config["dropout_prob"] >= 0.6:
            continue

        # hold all other parameters fixed
        key = tuple(r.config[x] for x in fixed_params)
        # construct generalization gap and accuracy lists (corresponding to each value of non-fixed parameter)

        run = r.history()
        del run['average_cross_entropy_over_epoch/test']
        # these are all NaN so not deleting this column results in dropna dropping all epochs
        run = run.dropna()
        # only keep epochs where we record the complexity measures
        run = run[[c for c in run.columns if c.lower()[:10] == "complexity"]]
        # construct list of runs (corresponding to each value of non-fixed parameter)
        run['gen'] = r.summary["generalization/error"]
        run['acc'] = r.summary["accuracy/test"]
        run[hp] = r.config[hp]

        run_dict[key].append(run)

    gen_correlation_dict = defaultdict(lambda: defaultdict(list))
    acc_correlation_dict = defaultdict(lambda: defaultdict(list))
    for key, group in run_dict.items():
        # only compute the correlation for epochs with values for each run
        # (some will end earlier so we will not have values for all runs for every recorded epoch)
        min_samples = min([len(r.index) for r in group])
        sample_measures = {}
        for i in range(min_samples - 1):
            runs_epoch_measures = []
            epoch = group[0].index[i]

            for run in group:
                runs_epoch_measures.append(run.loc[epoch])

            # with each epoch associate a list of the values of the measures at each point
            df = pd.DataFrame.from_records(runs_epoch_measures)
            df = df.astype(np.float64)
            sample_measures[epoch] = df

        runs_final_measures = []
        for run in group:
            runs_final_measures.append(run.iloc[-1])

        df = pd.DataFrame.from_records(runs_final_measures)
        df = df.astype(np.float64)
        sample_measures["final"] = df

        # Kendall's rank-correlation coefficients
        for k in sample_measures.keys():
            epoch_measures = sample_measures[k]
            for i in range(len(epoch_measures.columns)):
                gen_tau, _ = stats.kendalltau(epoch_measures['gen'], epoch_measures.iloc[:, i])
                gen_correlation_dict[epoch_measures.columns[i]][k].append(gen_tau)
                acc_tau, _ = stats.kendalltau(epoch_measures['acc'], epoch_measures.iloc[:, i])
                acc_correlation_dict[epoch_measures.columns[i]][k].append(acc_tau)

    for k, epochs in acc_correlation_dict.items():
        print("Measure: {}".format(k))
        for e, corrs in epochs.items():
            if len(corrs) < 3:
                continue
            print("Epoch: {}".format(e))
            print("Average correlation: {}".format(sum(corrs)/len(corrs)))
            print("Minimum correlation: {}".format(min(corrs, key=abs)))
            print("Correlations: ")
            print(corrs)

    save_fig(gen_correlation_dict, '{}/{}_gen_avg_fig'.format(hp, hp), plot_measures, "avg", 3)
    save_fig(acc_correlation_dict, '{}/{}_acc_avg_fig'.format(hp, hp), plot_measures, "avg", 3)
    save_fig(acc_correlation_dict, '{}/{}_acc_min_fig'.format(hp, hp), plot_measures, "min", 3)
    save_fig(gen_correlation_dict, '{}/{}_gen_min_fig'.format(hp, hp), plot_measures, "min", 3)


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


# get_kendall("lr", ["sotl", "sotl_10", "dist_spec_init_fft", "param_norm", "l2", "fro_over_spec_fft", "validation_acc", "fro_dist"])
# get_kendall("batch_size", ["sotl", "sotl_10", "dist_spec_init_fft", "param_norm", "l2", "fro_over_spec_fft", "validation_acc", "fro_dist"])
# get_kendall("model_depth", ["dist_spec_init_fft", "pacbayes_mag_flatness", "path_norm_over_margin", "fro_over_spec_fft", "validation_acc", "params", "sotl", "sotl_10"])
get_kendall("dropout_prob", ["dist_spec_init", "pacbayes_mag_flatness", "log_sum_of_fro", "sotl", "sotl_10", "validation_acc"])
