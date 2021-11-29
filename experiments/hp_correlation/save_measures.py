import wandb
import numpy as np
import pickle
import math
from typing import Tuple

api = wandb.Api()
project_names = ["rgm", "rgm_early_batches", "rgm_dropout"]
params = ["lr", "batch_norm", "batch_size", "model_depth", "dropout_prob"]
bad_runs = [(5, 0.3, 17), (5, 0.2, 17), (5, 0.15, 43), (5, 0.4, 43)]
reseed = {(5, 0.3, 42): 17, (5, 0.2, 16): 17, (5, 0.15, 42): 43, (5, 0.4, 42): 43}


def is_invalid_run(r):
    return r.state == "running" or r.config["dropout_prob"] * 10 % 1 != 0 or r.config["dropout_prob"] > 0.5 or \
           r.config["lr"] > 0.1 or r.config["lr"] < 0.001 or "batch_norm" not in r.config or not r.config["batch_norm"]


def save_runs():
    all_runs = {}
    num_runs = 0
    skipped = set()

    for proj_name in project_names:
        runs = api.runs(proj_name)
        for r in runs:
            if is_invalid_run(r):
                continue

            if "batch_norm_layers" in r.config:
                r.config["batch_norm"] = True if len(r.config["batch_norm_layers"]) > 0 else False
                r.config["model_depth"] = r.config["model_depth"]/3

            # some of the seeds got messed up in the dropout_prob experiments, fixing them here
            replace_key = (r.config["model_depth"], r.config["dropout_prob"], r.config["seed"])
            if replace_key in bad_runs:
                continue
            seed = r.config["seed"] = r.config["seed"] if replace_key not in reseed else reseed[replace_key]

            key = tuple(r.config[x] for x in params)
            if key not in all_runs:
                all_runs[key] = []

            # combine early batches and normal runs
            run = next((x for x in all_runs[key] if x.seed == seed), None)
            if run is None and (*key, seed) not in skipped:
                try:
                    run = Run(r)
                    all_runs[key].append(run)
                    num_runs += 1
                except ValueError:
                    print({x: r.config[x] for x in ["seed"] + params})
                    skipped.add((*key, seed))
            elif run is not None:
                try:
                    run.filter_steps(r.history(samples=10000, pandas=False))
                except ValueError:
                    print(run.seed)
                    print(run.config)
                    skipped.add((*key, seed))
                    all_runs[key].remove(run)
                    num_runs -= 1

    print("Total number of runs: {}".format(num_runs))
    print("{} runs skipped for containing invalid values".format(len(skipped)))

    with open("./data/runs.pickle", "wb+") as out:
        pickle.dump(all_runs, out)


class Run:
    def __init__(self, run):
        self.seed = run.config["seed"]
        self.config = {x: run.config[x] for x in params}
        self.steps_in_epoch = math.ceil(50000 / int(run.config["batch_size"]))
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
        for e in run_steps:
            step = e["_step"]

            if "complexity/SOTL" in e and step <= 1000 * self.steps_in_epoch - 1:
                e = self.filter_measures(e)
                if "Infinity" in e.values() or np.nan in e.values():
                    raise ValueError("Run contains invalid measure values")

                epoch = self.convert_step_to_epoch(step)
                e["epoch"] = epoch
                self.step_measures[step] = e
                if epoch.is_integer():
                    self.epoch_measures[int(epoch)] = e

                if step > self.final_step[0]:
                    self.final_step = (step, epoch)
                    self.final_test_acc = e["accuracy/test"]
                    self.final_gen_error = e["accuracy/train"] - e["accuracy/test"]

            if e["accuracy/test"] > self.best_test_acc:
                self.best_step = (step, self.convert_step_to_epoch(step))
                self.best_test_acc = e["accuracy/test"]
                self.best_gen_error = e["accuracy/train"] - e["accuracy/test"]

            if e["accuracy/test"] > 0.99 and step < self._99_step[0]:
                self._99_step = (step, self.convert_step_to_epoch(step))
                self._99_test_acc = e["accuracy/test"]
                self._99_gen_error = e["accuracy/train"] - e["accuracy/test"]

    @staticmethod
    def filter_measures(measures):
        save_cols = ["accuracy/train", "accuracy/test", "cross_entropy_epoch_end/train", "cross_entropy_epoch_end/test",
                     "_step"]
        filtered_measures = {}

        for k in measures:
            if k.lower()[:10] == "complexity":
                filtered_measures[k[11:]] = measures[k]
            elif k in save_cols:
                filtered_measures[k] = measures[k]
            elif k == "average_cross_entropy_over_epoch/train":
                filtered_measures["SOTL-1"] = measures[k]

        return filtered_measures

    def convert_step_to_epoch(self, step):
        return (step + 1) / self.steps_in_epoch


if __name__ == "__main__":
    save_runs()
