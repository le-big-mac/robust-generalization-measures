import wandb
import numpy as np
import pickle
import math
from typing import Tuple
from math import sqrt, log, log2, exp
from enum import Enum

api = wandb.Api()
project_names = ["rgm", "rgm_early_batches", "rgm_dropout"]
params = ["lr", "batch_norm", "batch_size", "model_depth", "dropout_prob"]
bad_runs = [(5, 0.3, 17), (5, 0.2, 17), (5, 0.4, 43)]
reseed = {(5, 0.3, 42): 17, (5, 0.2, 16): 17, (5, 0.4, 42): 43}
config_values_dict = {"no_dropout": {"lr": [0.001, 0.00158, 0.00316, 0.00631, 0.01, 0.02, 0.05, 0.1],
                                     "batch_size": [32, 64, 128, 256], "model_depth": [2, 3, 4, 5], "seed": [0, 17, 43],
                                     "dropout_prob": [0], "batch_norm": [True]},
                      "dropout": {"lr": [0.01], "batch_size": [32], "model_depth": [2, 3, 4, 5],
                                  "seed": [0, 17, 43], "dropout_prob": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                  "batch_norm": [True]},
                      "batch_norm": {"lr": [0.00316, 0.00631, 0.01, 0.02], "batch_size": [32],
                                     "model_depth": [1, 4/3, 5/3, 2], "seed": [0, 17, 43], "dropout_prob": [0],
                                     "batch_norm": [True, False]},
                      }

m = 50000
n = 32
C = 10
B = 115
abs_max_neg_margin = 2.2  # 99th percentile of negative margin values


class ExperimentType(Enum):
    NO_DROPOUT = 1
    DROPOUT = 2
    BATCH_NORM = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return TypeError


def make_combined(runs_config_dict):
    rd = runs_config_dict["dropout"]
    base_runs = {(x, ExperimentType.DROPOUT): rd[x] for x in rd}

    rnd = runs_config_dict["no_dropout"]
    for x in rnd:
        if x not in rd:
            base_runs[(x, ExperimentType.NO_DROPOUT)] = rnd[x]

    all_runs = base_runs.copy()

    rb = runs_config_dict["batch_norm"]
    for x in rb:
        all_runs[(x, ExperimentType.BATCH_NORM)] = rb[x]

    return all_runs, base_runs


def make_runs(name):
    config_range = config_values_dict[name]

    all_runs = {}
    num_runs = 0
    skipped = set()
    experiments = [ExperimentType[name.upper()]]

    for proj_name in project_names:
        runs = api.runs(proj_name)

        for r in runs:
            if name == "batch_norm" and "batch_norm_layers" in r.config:
                r.config["batch_norm"] = True if len(r.config["batch_norm_layers"]) > 0 else False
                r.config["model_depth"] = float(r.config["model_depth"])/3
            elif name == "batch_norm" or "batch_norm_layers" in r.config:
                continue

            # some of the seeds got messed up in the dropout_prob experiments, fixing them here
            replace_key = (r.config["model_depth"], r.config["dropout_prob"], r.config["seed"])
            if replace_key in bad_runs:
                continue
            seed = r.config["seed"] if replace_key not in reseed else reseed[replace_key]

            if not (all(r.config[hp] in config_range[hp] for hp in params) and seed in config_range["seed"]):
                continue

            key = tuple(r.config[x] for x in params)
            if key not in all_runs:
                all_runs[key] = []

            if name == "dropout" and r.config["dropout_prob"] == 0:
                experiments.append(ExperimentType.NO_DROPOUT)
            elif name == "no_dropout" and r.config["lr"] == 0.01 and r.config["batch_size"] == 32:
                experiments.append(ExperimentType.DROPOUT)

            # combine early batches and normal runs
            run = next((x for x in all_runs[key] if x.seed == seed), None)
            if run is None and (*key, seed) not in skipped:
                try:
                    run = Run(r, seed, tuple(experiments))
                    all_runs[key].append(run)
                    num_runs += 1
                except ValueError:
                    skipped.add((*key, seed))
            elif run is not None:
                try:
                    run.filter_steps(r.history(samples=10000, pandas=False))
                except ValueError:
                    skipped.add((*key, seed))
                    all_runs[key].remove(run)
                    num_runs -= 1

    print("Total number of runs: {}".format(num_runs))
    print("{} runs skipped for containing invalid values".format(len(skipped)))

    return all_runs


class Run:
    def __init__(self, run, seed, experiments):
        self.seed = seed
        self.experiments = experiments
        self.config = {x: run.config[x] for x in params}
        self.steps_in_epoch = math.ceil(m / int(run.config["batch_size"]))
        self.step_measures = {}
        self.epoch_measures = {}

        self.final_step: Tuple[int, int] = (-1, -1)
        self.final_test_acc = -1
        self.final_gen_error = -1

        self.best_step: Tuple[int, int] = (-1, -1)
        self.best_test_acc: int = -1
        self.best_gen_error: int = -1

        self._99_step: Tuple[int, int] = (np.inf, np.inf)
        self._99_test_acc: int = -1
        self._99_gen_error: int = -1

        self.filter_steps(run.history(samples=10000, pandas=False))

    def filter_steps(self, run_steps):
        temp_steps = {}

        for e in run_steps:
            step = e["_step"]
            temp_steps[step] = e.copy()
            epoch = self.convert_step_to_epoch(step)

            if "complexity/SOTL" in e and step <= 1000 * self.steps_in_epoch - 1:
                e = self.filter_measures(e)
                if np.nan in e.values():
                    raise ValueError("Run contains invalid measure values")

                e["epoch"] = epoch
                self.step_measures[step] = e
                if epoch.is_integer():
                    int_epoch = int(epoch)
                    self.epoch_measures[int_epoch] = e

                    sotl_ema = 0
                    for i in range(1, int_epoch):
                        sotl_ema += 0.9**(epoch - i) * float(temp_steps[self.convert_epoch_to_step(int_epoch - i)]["average_cross_entropy_over_epoch/train"])
                    e["sotl-ema"] = sotl_ema + e["sotl-1"]

                    default = {"average_cross_entropy_over_epoch/train": 0}
                    e["sotl-2"] = e["sotl-1"] + float(temp_steps.get(self.convert_epoch_to_step(int_epoch - 1), default)["average_cross_entropy_over_epoch/train"])
                    e["sotl-3"] = e["sotl-2"] + float(temp_steps.get(self.convert_epoch_to_step(int_epoch - 2), default)["average_cross_entropy_over_epoch/train"])
                    e["sotl-5"] = e["sotl-3"] + \
                                  float(temp_steps.get(self.convert_epoch_to_step(int_epoch - 3), default)["average_cross_entropy_over_epoch/train"]) + \
                                  float(temp_steps.get(self.convert_epoch_to_step(int_epoch - 4), default)["average_cross_entropy_over_epoch/train"])

                if step > self.final_step[0]:
                    self.final_step = (step, epoch)
                    self.final_test_acc = e["accuracy/test"]
                    self.final_gen_error = e["accuracy/train"] - e["accuracy/test"]

            if e["accuracy/test"] > self.best_test_acc:
                self.best_step = (step, epoch)
                self.best_test_acc = e["accuracy/test"]
                self.best_gen_error = e["accuracy/train"] - e["accuracy/test"]

            if e["accuracy/test"] > 0.99 and step < self._99_step[0]:
                self._99_step = (step, epoch)
                self._99_test_acc = e["accuracy/test"]
                self._99_gen_error = e["accuracy/train"] - e["accuracy/test"]

    def filter_measures(self, measures):
        save_cols = ["accuracy/train", "accuracy/test", "cross_entropy_epoch_end/train", "cross_entropy_epoch_end/test",
                     "_step"]
        discard = ["log_spec_init_main", "log_spec_orig_main", "log_prod_of_spec_over_margin", "log_prod_of_spec",
                   "fro_over_spec", "log_sum_of_spec_over_margin", "log_sum_of_spec", "dist_spec_init"]
        filtered_measures = {}

        for k in measures:
            if k.lower()[:10] == "complexity" and k.lower()[11:] not in discard:
                filtered_measures[k[11:].lower()] = float(measures[k])
            elif k in save_cols:
                filtered_measures[k] = float(measures[k])
            elif k == "average_cross_entropy_over_epoch/train":
                filtered_measures["sotl-1"] = float(measures[k])

        self.pacbayes_correction(filtered_measures)
        self.vc_dim(filtered_measures)
        self.new_margins(filtered_measures)
        self.spectral_complexity(filtered_measures)

        return filtered_measures

    def pacbayes_correction(self, measures):
        for t in ["", "_mag"]:
            inverse_sigma = sqrt(m) * measures["pacbayes{}_flatness".format(t)]

            for k in ["init", "orig"]:
                pacbayes_main = m * measures["pacbayes{}_{}".format(t, k)]**2
                pacbayes_main = 2 * (pacbayes_main - log(m * inverse_sigma) - 10)

                measures["pacbayes{}_{}_main".format(t, k)] = pacbayes_main
                measures["pacbayes{}_{}_full".format(t, k)] = \
                    measures["accuracy/train"] + 0.1 + sqrt((pacbayes_main + log(m * 100) + log(20000)) / (2 * (m-1)))

    def vc_dim(self, measures):
        # slightly inefficient to do for each epoch
        d = 3 * self.config["model_depth"] + 1

        measures["vc_dim"] = \
            measures["accuracy/train"] + \
            3888 * C * sqrt(d * log2(6 * d * n)**3 * measures["params"]**2) + sqrt(log(100) / m)

    def new_margins(self, measures):
        log_prod_measures = ["log_prod_of_spec_over{}_fft", "log_spec_init_main{}_fft",
                             "log_spec_orig_main{}_fft", "log_prod_of_fro_over{}"]
        log_sum_measures = ["log_sum_of_spec_over{}_fft", "log_sum_of_fro_over{}"]
        prod_measures = ["path_norm_over{}"]

        inverse_margin = sqrt(m) * measures["inverse_margin"] if measures["accuracy/train"] >= 0.9 \
            else - sqrt(m) * measures["inverse_margin"]
        measures["margin"] = 1 / inverse_margin

        max_margin_over_margin = inverse_margin * abs_max_neg_margin
        try:
            exponential_margin = exp(-(1/inverse_margin))
        except OverflowError:
            exponential_margin = np.inf

        new_margins = {"pos_margin": np.inf if inverse_margin < 0 else measures["inverse_margin"],
                       "shifted_margin":
                           max_margin_over_margin / (max_margin_over_margin + 1)
                           if 1 / inverse_margin > -2.2 else
                           1 / abs(max_margin_over_margin) * 150,
                       "exponential_margin": exponential_margin,
                       "normalized_exponential_margin": exponential_margin/exp(abs_max_neg_margin)  # 99% in range 0-1
                       }

        for k, mgn in new_margins.items():
            measures[k] = mgn

            if k == "pos_margin":
                for m_name in log_prod_measures + log_sum_measures + prod_measures:
                    if mgn == np.inf:
                        measures[m_name.format("_" + k)] = np.inf
                    else:
                        measures[m_name.format("_" + k)] = \
                            measures.get(m_name.format("_margin")) or measures.get(m_name.format(""))

            else:
                old_log_margin_correction = - log(abs(inverse_margin))  # = log(margin)
                if k == "exponential_margin":
                    new_log_margin_correction = 1/2 * - 1 / inverse_margin
                elif k == "normalized_exponential_margin":
                    new_log_margin_correction = 1/2 * (- 1 / inverse_margin - abs_max_neg_margin)
                else:
                    new_log_margin_correction = 1/2 * log(mgn)  # = - log(sqrt(new_margin))

                for m_name in log_prod_measures:
                    old_meas = measures.get(m_name.format("_margin")) or measures.get(m_name.format(""))
                    if type(old_meas) == str:
                        print(old_meas)
                    measures[m_name.format("_" + k)] = \
                        old_meas + old_log_margin_correction + new_log_margin_correction

                for m_name in log_sum_measures:
                    old_meas = measures[m_name.format("_margin")]
                    measures[m_name.format("_" + k)] = \
                        old_meas + old_log_margin_correction / (3 * self.config["model_depth"] + 1) + \
                        new_log_margin_correction / (3 * self.config["model_depth"] + 1)

                for m_name in prod_measures:
                    old_meas = measures[m_name.format("_margin")]
                    measures[m_name.format("_" + k)] = old_meas * sqrt(mgn)/inverse_margin

    def spectral_complexity(self, measures):
        depth_const = 84 * B * (5 * self.config["model_depth"] * sqrt(200) + sqrt(10)) + \
                      sqrt(log(4 * n**2 * (3 * self.config["model_depth"] + 1)))

        for k in ["init", "orig"]:
            spec_main = exp(2 * measures["log_spec_{}_main_fft".format(k)])
            spec_full = 0.1 + sqrt(depth_const * spec_main + log(100 * m) * measures["inverse_margin"]**2)

            measures["spec_{}".format(k)] = spec_full
            measures["spec_{}_pos_margin".format(k)] = spec_full if measures["pos_margin"] < np.inf else np.inf

            for m_name in ["shifted_margin", "exponential_margin", "normalized_exponential_margin"]:
                inverse_margin = measures[m_name] / m

                try:
                    spec_main = exp(2 * measures["log_spec_{}_main_{}_fft".format(k, m_name)])
                except OverflowError:
                    spec_main = np.inf

                measures["spec_{}_{}".format(k, m_name)] = \
                    0.1 + sqrt(depth_const * spec_main + log(100 * m) * inverse_margin)

    def convert_step_to_epoch(self, step):
        return (step + 1) / self.steps_in_epoch

    def convert_epoch_to_step(self, epoch):
        return epoch * self.steps_in_epoch - 1


if __name__ == "__main__":
    cleaned_runs = {x: make_runs(x) for x in config_values_dict}
    for x in cleaned_runs:
        with open("./data/new/runs_{}.pickle".format(x), "wb") as f:
            pickle.dump(cleaned_runs[x], f)

    all_runs, base_runs = make_combined(cleaned_runs)
    with open("./data/new/runs_all.pickle", "wb") as f:
        pickle.dump(all_runs, f)
    with open("./data/new/runs_base.pickle", "wb") as f:
        pickle.dump(base_runs, f)
