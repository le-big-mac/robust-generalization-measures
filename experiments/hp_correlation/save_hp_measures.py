import wandb
import pandas as pd
from collections import defaultdict
import numpy as np
import sys
import pickle

api = wandb.Api()
runs = api.runs("rgm")
hparams = ['lr', 'seed', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob']
# TODO: fix config values for dropout
config_values_dict = {"lr": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                             'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43],
                             'dropout_prob': [0], 'batch_norm': [True]},
                      "model_depth": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                                      'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5],
                                      'seed': [0, 17, 43], 'dropout_prob': [0], 'batch_norm': [True]},
                      "batch_size": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                                     'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43],
                                     'dropout_prob': [0], 'batch_norm': [True]},
                      "dropout": {'lr': [0.01], 'batch_size': [32], 'model_depth': [2, 3, 4], 'seed': [0, 17, 43],
                                  'dropout_prob': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 'batch_norm': [True]},
                      "batch_norm": {'lr': [0.00316, 0.00631, 0.01, 0.02], 'batch_size': [32],
                                     'model_depth': [3, 4, 5, 6], 'seed': [0, 17, 43], 'dropout_prob': [0],
                                     'batch_norm': [0, 1]}
                      }


def save_hp_measures(hp: str):
    fixed_params = hparams
    fixed_params.remove(hp)
    config_range = config_values_dict[hp]

    run_dict = defaultdict(list)
    for r in runs:
        # if r.state == "running" \
        #         or "batch_norm" not in r.config.keys() \
        #         or r.config["batch_norm"] not in config_range["batch_norm"] \
        #         or r.config["dropout_prob"] not in config_range["dropout_prob"] \
        #         or r.config["lr"] not in config_range["lr"] \
        #         or r.config["seed"] not in config_range["seed"] \
        #         or r.config["batch_size"] not in config_range["batch_size"] \
        #         or r.config["model_depth"] not in config_range["model_depth"]:
        #     continue

        if r.state == "running" \
                or "batch_norm_layers" not in r.config.keys() \
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
        run = run[[c for c in run.columns if c.lower()[:10] == "complexity" or c.lower() == "accuracy/train"
                   or c.lower() == "cross_entropy_epoch_end/train"]]
        if run.index[0] == 0:
            run.index += 1

        # add in value for variable hyperparameter and gen error and test accuracy
        run['gen'] = r.summary["generalization/error"]
        run['acc'] = r.summary["accuracy/test"]
        run[hp] = 1 if len(r.config["batch_norm_layers"]) > 0 else 0

        # construct list of runs (corresponding to each value of non-fixed parameter)
        run_dict[key].append(run)

    frame_dict = {}

    for key, group in run_dict.items():
        # only use keys that have a value for every value of the hyperparameter
        if len(group) != len(config_range[hp]):
            print("Key: {}".format(key))
            continue

        num_samples, loc = max([(len(group[r].index), r) for r in range(len(group))])
        sample_measures = {}

        for i in range(num_samples - 1):
            runs_epoch_measures = []
            epoch = group[loc].index[i]

            for run in group:
                try:
                    measures = run.loc[epoch].copy(deep=True)
                except KeyError:
                    measures = run.iloc[-1].copy(deep=True)
                    measures[:] = np.nan
                runs_epoch_measures.append(measures)

            # with each epoch associate a list of the values of the measures at each point
            df = pd.DataFrame.from_records(runs_epoch_measures)
            df = df.astype(np.float64)
            sample_measures[epoch] = df

        runs_final_measures = []
        for run in group:
            runs_final_measures.append(run.iloc[-1].copy(deep=True))

        df = pd.DataFrame.from_records(runs_final_measures)
        df = df.astype(np.float64)
        # ensure final values are sorted to end
        sample_measures[sys.maxsize] = df

        frame_dict[key] = sample_measures

    with open("./results/{}/measures.pickle".format(hp), "wb+") as out:
        pickle.dump(frame_dict, out)


save_hp_measures("batch_norm")
