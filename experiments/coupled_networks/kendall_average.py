import wandb
import pandas as pd
import scipy.stats as stats
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys

api = wandb.Api()
runs = api.runs("rgm")
hparams = ['lr', 'seed', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob']
config_params = ['lr', 'batch_size', 'model_depth', 'seed']
results_order = ["PARAMS", "INVERSE_MARGIN", "LOG_SPEC_INIT_MAIN", "LOG_SPEC_INIT_MAIN_FFT", "LOG_SPEC_ORIG_MAIN",
                 "LOG_SPEC_ORIG_MAIN_FFT", "LOG_PROD_OF_SPEC_OVER_MARGIN", "LOG_SUM_OF_SPEC_OVER_MARGIN_FFT",
                 "LOG_PROD_OF_SPEC", "LOG_PROD_OF_SPEC_FFT", "LOG_SUM_OF_SPEC_OVER_MARGIN",
                 "LOG_SUM_OF_SPEC_OVER_MARGIN_FFT", "LOG_SUM_OF_SPEC", "LOG_SUM_OF_SPEC_FFT", "DIST_SPEC_INIT",
                 "DIST_SPEC_INIT_FFT", "FRO_OVER_SPEC", "FRO_OVER_SPEC_FFT", "LOG_PROD_OF_FRO_OVER_MARGIN",
                 "LOG_PROD_OF_FRO", "LOG_SUM_OF_FRO_OVER_MARGIN", "LOG_SUM_OF_FRO", "FRO_DIST", "PARAM_NORM",
                 "PATH_NORM_OVER_MARGIN", "PATH_NORM", "PACBAYES_INIT", "PACBAYES_ORIG", "PACBAYES_FLATNESS",
                 "PACBAYES_MAG_INIT", "PACBAYES_MAG_ORIG", "PACBAYES_MAG_FLATNESS", "SOTL", "SOTL_10", "L2", "L2_DIST",
                 "VALIDATION_ACC"]


def get_kendall(hp: str, config_range: dict):
    fixed_params = hparams
    fixed_params.remove(hp)

    run_dict = defaultdict(list)
    for r in runs:
        if r.state == "running" \
                or r.config["batch_norm"] not in config_range["batch_norm"] \
                or r.config["dropout_prob"] not in config_range["dropout_prob"] \
                or r.config["lr"] not in config_range["lr"] \
                or r.config["seed"] not in config_range["seed"] \
                or r.config["batch_size"] not in config_range["batch_size"] \
                or r.config["model_depth"] not in config_range["model_depth"]:
            continue

        # TODO: switch out bad seeds
        # hold all other parameters fixed
        key = tuple(r.config[x] for x in fixed_params)
        # construct generalization gap and accuracy lists (corresponding to each value of non-fixed parameter)

        run = r.history()
        del run['average_cross_entropy_over_epoch/test']
        # these are all NaN so not deleting this column results in dropna dropping all epochs
        run = run.dropna()
        # only keep epochs where we record the complexity measures
        run = run[[c for c in run.columns if c.lower()[:10] == "complexity" or c.lower() == "accuracy/train"]]
        if run.index[0] == 1:
            run.index -= 1

        # add in value for variable hyperparameter and gen error and test accuracy
        run['gen'] = r.summary["generalization/error"]
        run['acc'] = r.summary["accuracy/test"]
        run[hp] = r.config[hp]

        # construct list of runs (corresponding to each value of non-fixed parameter)
        run_dict[key].append(run)

    gen_correlation_dict = defaultdict(lambda: defaultdict(list))
    acc_correlation_dict = defaultdict(lambda: defaultdict(list))
    for key, group in run_dict.items():
        # only use keys that have a value for every value of the hyperparameter
        if len(group) != len(config_values[hp]):
            print("Key: {}".format(key))
            continue

        # only compute the correlation for epochs with values for each run
        # (some will end earlier so we will not have values for all runs for every recorded epoch)
        num_samples, loc = min([(len(group[r].index), r) for r in range(len(group))])
        # TODO: use final values for epochs after training ends
        # num_samples, loc = max([(len(group[r].index), r) for r in range(len(group))])
        sample_measures = {}

        for i in range(num_samples - 1):
            runs_epoch_measures = []
            epoch = group[loc].index[i]

            for run in group:
                try:
                    measures = run.loc[epoch]
                except KeyError:
                    measures = run.iloc[-1]
                runs_epoch_measures.append(measures)

            # with each epoch associate a list of the values of the measures at each point
            df = pd.DataFrame.from_records(runs_epoch_measures)
            df = df.astype(np.float64)
            sample_measures[epoch] = df

        runs_final_measures = []
        for run in group:
            runs_final_measures.append(run.iloc[-1])

        df = pd.DataFrame.from_records(runs_final_measures)
        df = df.astype(np.float64)
        # ensure final values are sorted to end
        sample_measures[sys.maxsize] = df

        # Kendall's rank-correlation coefficients
        for k in sample_measures.keys():
            epoch_measures = sample_measures[k]
            for i in range(len(epoch_measures.columns)):
                gen_tau, _ = stats.kendalltau(epoch_measures['gen'], epoch_measures.iloc[:, i])
                gen_correlation_dict[epoch_measures.columns[i]][k].append(gen_tau)
                acc_tau, _ = stats.kendalltau(epoch_measures['acc'], epoch_measures.iloc[:, i])
                acc_correlation_dict[epoch_measures.columns[i]][k].append(acc_tau)

        # TODO: append final values of correlations that end earlier

    with open("./results/{}_new.csv".format(hp), "w+") as csv_file:

        for k in results_order:
            epochs = acc_correlation_dict["complexity/{}".format(k)]
            print("Measure: {} \n".format(k))
            csv_str = "{}, ".format(k)

            for e, corrs in sorted(epochs.items()):
                # only include epochs for which we have correlation values for all runs
                if len(corrs) < len(epochs[0]):
                    continue
                csv_str += make_stats_string(corrs)
                print("Epoch: {}".format(e))
                print("Average correlation: {}".format(sum(corrs)/len(corrs)))
                print("Minimum correlation: {}".format(min(corrs, key=abs)))
                print("Variance: {}".format(np.var(corrs)))
                print("Median: {}".format(np.median(corrs)))
                print("25%: {}".format(np.quantile(corrs, 0.25)))
                print("75%: {} \n".format(np.quantile(corrs, 0.75)))

            csv_file.write(csv_str)

        csv_str = "TRAIN_ACC, "
        epochs = acc_correlation_dict["accuracy/train"]
        for e, corrs in sorted(epochs.items()):
            if len(corrs) < len(epochs[0]):
                continue
            csv_str += make_stats_string(corrs)
        csv_file.write(csv_str)

        csv_str = "{}, ".format(hp)
        epochs = acc_correlation_dict[hp]
        for e, corrs in sorted(epochs.items()):
            if len(corrs) < len(epochs[0]):
                continue
            csv_str += make_stats_string(corrs)
        csv_file.write(csv_str)

    # save_fig(gen_correlation_dict, '{}/{}_gen_avg_fig'.format(hp, hp), plot_measures, "avg", 3)
    # save_fig(acc_correlation_dict, '{}/{}_acc_avg_fig'.format(hp, hp), plot_measures, "avg", 3)
    # save_fig(acc_correlation_dict, '{}/{}_acc_min_fig'.format(hp, hp), plot_measures, "min", 3)
    # save_fig(gen_correlation_dict, '{}/{}_gen_min_fig'.format(hp, hp), plot_measures, "min", 3)


def make_stats_string(corrs):
    csv_str = "{}, ".format(sum(corrs) / len(corrs))
    csv_str += "{}, ".format(min(corrs, key=abs))
    csv_str += "{}, ".format(np.var(corrs))
    csv_str += "{}, ".format(np.median(corrs))
    csv_str += "{}, ".format(np.quantile(corrs, 0.25))
    csv_str += "{}, ".format(np.quantile(corrs, 0.75))
    csv_str += "\n"

    return csv_str


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


config_values = {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1], 'batch_size': [32, 64, 128, 256],
                 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43], "dropout_prob": [0],
                 'batch_norm': [True]}
# config_values = {'lr': [0.01], 'batch_size': [32],
#                  'model_depth': [2, 3, 4], 'seed': [0, 17, 43], "dropout_prob": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#                  'batch_norm': [True]}
get_kendall("lr", config_values)
