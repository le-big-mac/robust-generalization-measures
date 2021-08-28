import wandb
import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import math

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
                      "dropout": {'lr': [0.01], 'batch_size': [32], 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43],
                                  'dropout_prob': [0, 0.05, 0.1], 'batch_norm': [True]},
                      "batch_norm": {'lr': [0.00316, 0.00631, 0.01, 0.02], 'batch_size': [32],
                                     'model_depth': [3, 4, 5, 6], 'seed': [0, 17, 43], 'dropout_prob': [0],
                                     'batch_norm': [0, 1]}
                      }


def save_hp_measures(hp: str, epochs):
    fixed_params = hparams
    fixed_params.remove(hp)
    config_range = config_values_dict[hp]

    run_dict = defaultdict(list)
    for r in runs:
        if r.state == "running" \
                or "batch_norm" not in r.config.keys() \
                or r.config["batch_norm"] not in config_range["batch_norm"] \
                or r.config["dropout_prob"] not in config_range["dropout_prob"] \
                or r.config["lr"] not in config_range["lr"] \
                or r.config["seed"] not in config_range["seed"] \
                or r.config["batch_size"] not in config_range["batch_size"] \
                or r.config["model_depth"] not in config_range["model_depth"]:
            continue

        # if r.state == "running" \
        #         or "batch_norm_layers" not in r.config.keys() \
        #         or r.config["dropout_prob"] not in config_range["dropout_prob"] \
        #         or r.config["lr"] not in config_range["lr"] \
        #         or r.config["seed"] not in config_range["seed"] \
        #         or r.config["batch_size"] not in config_range["batch_size"] \
        #         or r.config["model_depth"] not in config_range["model_depth"]:
        #     continue

        # TODO: switch out bad seeds
        # hold all other parameters fixed
        key = tuple(r.config[x] for x in fixed_params)
        len_loader = math.ceil(50000/int(r.config["batch_size"]))
        epoch_steps = [i * len_loader - 1 for i in epochs]

        def transform_step(step):
            return int((step + 1) / len_loader)

        print("Key: {}".format(key))
        print("{}".format(r.config[hp]))

        history = r.scan_history()
        run = []
        final = {"_step": -1}
        for e in history:
            if e["_step"] in epoch_steps:
                e = e.copy()
                e["epoch"] = transform_step(e["_step"])
                run.append(e)
            if "complexity/SOTL" in e and e["_step"] > final["_step"]:
                e = e.copy()
                e["epoch"] = "final"
                final = e

        run.append(final)
        run = pd.DataFrame(run)
        run = run.set_index("epoch")
        assert not run["complexity/SOTL"].isnull().values.any()

        run = run[[c for c in run.columns if c.lower()[:10] == "complexity" or c.lower() == "accuracy/train"
                   or c.lower() == "accuracy/test" or c.lower() == "cross_entropy_epoch_end/train"
                   or c.lower() == "cross_entropy_epoch_end/test"]]

        # add in value for variable hyperparameter
        run["hp"] = r.config[hp]
        # run[hp] = 1 if len(r.config["batch_norm_layers"]) > 0 else 0

        run["final_gen_error"] = run.loc["final"]["accuracy/train"] - run.loc["final"]["accuracy/test"]
        run["final_test_acc"] = run.loc["final"]["accuracy/test"]

        # TODO: this commented out code allows us to get best and 99 measures at epochs in which they are defined (multiples of 5)
        # # use best test performance stopping criterion
        # idx_best = run["accuracy/test"].idxmax()
        # best_measures = run.loc[idx_best]
        # best_measures.name = "best"
        # run["best_gen_error"] = best_measures["accuracy/train"] - best_measures["accuracy/test"]
        # run["best_test_acc"] = run["accuracy/test"][idx_best]
        #
        # # use 99 train accuracy stopping criterion (if we want the measures may need to ensure epoch is multiple of 5)
        # idx_99 = (run["accuracy/train"] > 0.99).idxmax()
        # if idx_99 >= 1:
        #     _99_measures = run.loc[idx_99]
        #     _99_measures.name = "99 acc"
        #     run["99_gen_error"] = _99_measures["accuracy/train"] - _99_measures["accuracy/test"]
        #     run["99_test_acc"] = run["accuracy/test"][idx_99]
        # else:
        #     run["99_gen_error"] = np.nan
        #     run["99_test_acc"] = np.nan

        accs = list(r.scan_history(keys=["accuracy/train", "accuracy/test", "_step"]))

        best = max(accs, key=lambda x: x["accuracy/train"])
        run["best_gen_error"] = best["accuracy/train"] - best["accuracy/test"]
        run["best_test_acc"] = best["accuracy/test"]

        above_99 = [i for i in accs if i["accuracy/train"] > 0.99]
        if len(above_99) == 0:
            run["99_gen_error"] = np.nan
            run["99_test_acc"] = np.nan
        else:
            first_99 = min(above_99, key=lambda x: x["_step"])
            run["99_gen_error"] = first_99["accuracy/train"] - first_99["accuracy/test"]
            run["99_test_acc"] = first_99["accuracy/test"]

        print(run)

        # construct list of runs (corresponding to each value of non-fixed parameter)
        run_dict[key].append(run)

    fixed_params_dict = {}

    for key, group in run_dict.items():
        # only use keys that have a value for every value of the hyperparameter
        if len(group) != len(config_range[hp]):
            print("Key: {}".format(key))
            continue

        num_samples, longest_run = max([(len(group[r].index), r) for r in range(len(group))])
        sample_measures = {}

        for epoch in epochs:
            runs_epoch_measures = []

            for run in group:
                try:
                    measures = run.loc[epoch].copy(deep=True)
                except KeyError:
                    measures = run.iloc[-1].copy(deep=True)
                    print("Key: {}".format(key))
                    print("HP: {}".format(measures["hp"]))
                    print("Epoch: {}".format(epoch))
                    print(run)
                    measures[:] = np.nan
                runs_epoch_measures.append(measures)

            # with each epoch associate a dataframe of the values of the measures for different hps
            df = pd.DataFrame(runs_epoch_measures)
            print(df)
            df = df.astype(np.float64)
            df = df.set_index("hp")
            sample_measures[epoch] = df

        runs_final_measures = []
        for run in group:
            runs_final_measures.append(run.loc["final"].copy(deep=True))

        df = pd.DataFrame.from_records(runs_final_measures)
        df = df.astype(np.float64)
        df = df.set_index("hp")
        # ensure final values are sorted to end
        sample_measures["final"] = df

        fixed_params_dict[key] = sample_measures

    with open("./results/{}/measures_{}.pickle".format(hp, epochs), "wb+") as out:
        pickle.dump(fixed_params_dict, out)


def save_all_measures(config_range):
    run_dict = {}
    for r in runs:
        if r.state == "running" \
                or "batch_norm" not in r.config.keys() \
                or r.config["batch_norm"] not in config_range["batch_norm"] \
                or r.config["dropout_prob"] not in config_range["dropout_prob"] \
                or r.config["lr"] not in config_range["lr"] \
                or r.config["seed"] not in config_range["seed"] \
                or r.config["batch_size"] not in config_range["batch_size"] \
                or r.config["model_depth"] not in config_range["model_depth"]:
            continue

        key = tuple(r.config[x] for x in hparams)
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

        # construct list of runs (corresponding to each value of non-fixed parameter)
        run_dict[key] = run

    with open("./results/all/measures_aah.pickle", "wb+") as out:
        pickle.dump(run_dict, out)


config_range = {'lr': [0.001, 0.00316, 0.01, 0.02], 'batch_size': [32, 64, 128, 256],
                'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43], 'dropout_prob': [0], 'batch_norm': [True]}
save_hp_measures('lr', [1, 5, 10, 15, 20])
