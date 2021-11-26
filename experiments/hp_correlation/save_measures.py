import wandb
import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import math

api = wandb.Api()
runs = api.runs("rgm")
hparams = ['lr', 'seed', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob']
config_values_dict = {"lr": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                             'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43],
                             'dropout_prob': [0], 'batch_norm': [True]},
                      "model_depth": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                                      'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5],
                                      'seed': [0, 17, 43], 'dropout_prob': [0], 'batch_norm': [True]},
                      "batch_size": {'lr': [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                                     'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5], 'seed': [0, 17, 43],
                                     'dropout_prob': [0], 'batch_norm': [True]},
                      "dropout_prob": {'lr': [0.01], 'batch_size': [32], 'model_depth': [2, 3, 4, 5],
                                       'seed': [0, 16, 17, 42, 43],
                                       'dropout_prob': [0, 0.1, 0.2, 0.3, 0.4, 0.5], 'batch_norm': [True]},
                      "batch_norm": {'lr': [0.00316, 0.00631, 0.01, 0.02], 'batch_size': [32],
                                     'model_depth': [3, 4, 5, 6], 'seed': [0, 17, 43], 'dropout_prob': [0],
                                     'batch_norm': [True, False]},
                      "restricted_lr": {'lr': [0.001, 0.00316, 0.01, 0.05],
                                        'batch_size': [32, 64, 128, 256], 'model_depth': [2, 3, 4, 5],
                                        'seed': [0, 17, 43], 'dropout_prob': [0], 'batch_norm': [True]},
                      }
bad_runs = [(5, 0.3, 17), (5, 0.2, 17), (5, 0.15, 43)]
reseed = {(5, 0.3, 42): 17, (5, 0.2, 16): 17, (5, 0.15, 42): 43}


def save_runs(epochs, config_range, name):
    all_runs = []

    num_runs = 0
    for r in runs:
        if name == "batch_norm" and "batch_norm_layers" not in r.config:
            continue
        elif name == "batch_norm":
            r.config["batch_norm"] = True if len(r.config["batch_norm_layers"]) > 0 else False
            r.config["model_depth"] = r.config["model_depth"]/3

        is_valid = all(r.config[key] in config_range[key] for key in hparams)

        if r.state == "running" or "batch_norm" not in r.config or not is_valid:
            continue

        # some of the seeds got messed up in the dropout_prob experiments, fixing them here
        replace_key = (r.config["model_depth"], r.config["dropout_prob"], r.config["seed"])
        if replace_key in bad_runs:
            continue
        r.config["seed"] = r.config["seed"] if replace_key not in reseed else reseed[replace_key]

        config = {x: r.config[x] for x in hparams}
        run = format_run(r, epochs)

        # construct list of runs (corresponding to each value of non-fixed parameter)
        if "Infinity" not in run.values and np.nan not in run.values:
            all_runs.append((config, run))
            num_runs += 1
        else:
            print(config)

    print(num_runs)

    with open("./data/runs_{}_{}.pickle".format(name, epochs), "wb+") as out:
        pickle.dump(all_runs, out)


def weighted_hparams_save_runs(epochs, config_range, name):
    all_runs = {}

    num_runs = 0
    for r in runs:
        if name == "batch_norm" and "batch_norm_layers" not in r.config:
            continue
        elif name == "batch_norm":
            r.config["batch_norm"] = True if len(r.config["batch_norm_layers"]) > 0 else False
            r.config["model_depth"] = r.config["model_depth"]/3

        is_valid = all(r.config[key] in config_range[key] for key in hparams)

        if r.state == "running" or "batch_norm" not in r.config or not is_valid:
            continue

        # some of the seeds got messed up in the dropout_prob experiments, fixing them here
        replace_key = (r.config["model_depth"], r.config["dropout_prob"], r.config["seed"])
        if replace_key in bad_runs:
            continue
        r.config["seed"] = r.config["seed"] if replace_key not in reseed else reseed[replace_key]

        params = ['lr', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob']
        key = tuple(r.config[x] for x in params)
        if key not in all_runs:
            all_runs[key] = []

        config = {x: r.config[x] for x in hparams}
        run = format_run(r, epochs)

        # construct list of runs (corresponding to each value of non-fixed parameter)
        if "Infinity" not in run.values and np.nan not in run.values:
            all_runs[key].append((config, run))
            num_runs += 1
        else:
            print(config)

    print(num_runs)

    with open("./data/runs_joined_seed_{}_{}.pickle".format(name, epochs), "wb+") as out:
        pickle.dump(all_runs, out)


def save_hp_measures(hp: str, epochs):
    fixed_params = hparams.copy()
    fixed_params.remove(hp)
    config_range = config_values_dict[hp]

    run_dict = defaultdict(list)
    num_runs = 0
    for r in runs:
        if hp == "batch_norm" and "batch_norm_layers" not in r.config:
            continue
        elif hp == "batch_norm":
            r.config["batch_norm"] = True if len(r.config["batch_norm_layers"]) > 0 else False

        if r.state == "running" \
                or "batch_norm" not in r.config \
                or r.config["batch_norm"] not in config_range["batch_norm"] \
                or r.config["dropout_prob"] not in config_range["dropout_prob"] \
                or r.config["lr"] not in config_range["lr"] \
                or r.config["seed"] not in config_range["seed"] \
                or r.config["batch_size"] not in config_range["batch_size"] \
                or r.config["model_depth"] not in config_range["model_depth"]:
            continue

        # some of the seeds got messed up in the dropout_prob experiments, fixing them here
        replace_key = (r.config["model_depth"], r.config["dropout_prob"], r.config["seed"])
        if replace_key in bad_runs:
            continue
        r.config["seed"] = r.config["seed"] if replace_key not in reseed else reseed[replace_key]
        num_runs += 1

        key = tuple(r.config[x] for x in fixed_params)

        run = format_run(r, epochs)
        run["hp"] = r.config[hp]

        # construct list of runs (corresponding to each value of non-fixed parameter)
        run_dict[key].append(run)

    print("Num_runs: {}".format(num_runs))
    fixed_params_dict = {}

    for key, group in run_dict.items():
        # only use keys that have a value for every value of the hyperparameter
        bad_val = True in ["Infinity" in x.values for x in group]
        if len(group) != len(config_range[hp]) or bad_val:
            print("Key: {}".format(key))
            continue

        sample_measures = {}

        for epoch in epochs:
            runs_epoch_measures = []

            for run in group:
                try:
                    measures = run.loc[epoch].copy(deep=True)
                except KeyError:
                    measures = run.iloc[-1].copy(deep=True)
                    # print("Key: {}".format(key))
                    # print("HP: {}".format(measures["hp"]))
                    # print("Epoch: {}".format(epoch))
                    # print(run)
                    local_hp = measures["hp"]
                    measures[:] = np.nan
                    measures["hp"] = local_hp
                runs_epoch_measures.append(measures)

            # with each epoch associate a dataframe of the values of the measures for different hps
            df = pd.DataFrame(runs_epoch_measures)
            df = df.astype(np.float64)
            df = df.set_index("hp")
            sample_measures[epoch] = df

        runs_final_measures = []
        for run in group:
            runs_final_measures.append(run.loc["final"].copy(deep=True))

        df = pd.DataFrame.from_records(runs_final_measures)
        df = df.astype(np.float64)
        df = df.set_index("hp")
        sample_measures["final"] = df

        fixed_params_dict[key] = sample_measures

    with open("./results/{}/measures_{}.pickle".format(hp, epochs), "wb+") as out:
        pickle.dump(fixed_params_dict, out)


def format_run(r, epochs):
    len_loader = math.ceil(50000 / int(r.config["batch_size"]))
    epoch_steps = [i * len_loader - 1 for i in epochs]

    def transform_step(step):
        return int((step + 1) / len_loader)

    history = r.history(samples=10000, pandas=False)
    run = []
    final = {"_step": -1}
    for e in history:
        if e["_step"] in epoch_steps:
            e = e.copy()
            e["epoch"] = transform_step(e["_step"])
            run.append(e)
        if "complexity/SOTL" in e and final["_step"] < e["_step"] <= 1000 * len_loader - 1:
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

    run["final_gen_error"] = run.loc["final"]["accuracy/train"] - run.loc["final"]["accuracy/test"]
    run["final_test_acc"] = run.loc["final"]["accuracy/test"]

    best = max(history, key=lambda x: x["accuracy/train"])
    run["best_gen_error"] = best["accuracy/train"] - best["accuracy/test"]
    run["best_test_acc"] = best["accuracy/test"]

    above_99 = [i for i in history if i["accuracy/train"] > 0.99]
    if len(above_99) == 0:
        run["99_gen_error"] = np.nan
        run["99_test_acc"] = np.nan
    else:
        first_99 = min(above_99, key=lambda x: x["_step"])
        run["99_gen_error"] = first_99["accuracy/train"] - first_99["accuracy/test"]
        run["99_test_acc"] = first_99["accuracy/test"]

    return run


if __name__ == "__main__":
    c = config_values_dict["batch_norm"].copy()
    c["model_depth"] = [x/3 for x in c["model_depth"]]
    # save_runs([1, 5, 10, 15, 20], c, "basic")
    # save_hp_measures("lr", [1, 5, 10, 15, 20])
    weighted_hparams_save_runs([1, 5, 10, 15, 20], c, "batch_norm")
