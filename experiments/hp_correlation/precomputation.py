import pickle
import numpy as np
from save_measures import Run, ExperimentType
from tqdm import tqdm
from itertools import islice

stop_types = ["final", "best", "_99"]
early_batches = (0, 1, 10, 100)


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def sign_error(c1, c2, g1, g2):
    return np.sign(c1 - c2) * np.sign(g1 - g2)


def hoeffding_weight(delta_gen, m=10000):
    """
    This value has the following guarantee. If your measurement of the generalization gap is computed
    using m (say m=10,000) independent samples, then accepting samples only when this value > p would
    mean that the difference of the two generalization estimates has the correct sign with probability >= p
    """
    phi = 2 * np.exp(-2 * m * (np.abs(delta_gen) / 2) ** 2)

    return max(0., 1. - phi) ** 2


class RunPair:
    def __init__(self, run1, run2):
        for st in stop_types:
            setattr(self, "{}_weight".format(st), hoeffding_weight(getattr(run1, "{}_test_acc".format(st)) -
                                                                   getattr(run2, "{}_test_acc".format(st))))
            setattr(self, "early_batches_{}_error".format(st), {})
            setattr(self, "epochs_{}_error".format(st), {})

        for st in stop_types:
            g1 = getattr(run1, "{}_test_acc".format(st))
            g2 = getattr(run2, "{}_test_acc".format(st))

            if ExperimentType.BATCH_NORM not in run1.experiments \
                    and ExperimentType.BATCH_NORM not in run2.experiments:
                for step in early_batches:
                    error_dict = getattr(self, "early_batches_{}_error".format(st))[step] = {}

                    for m in run1.step_measures[step]:
                        c1 = run1.step_measures[step][m]
                        c2 = run1.step_measures[step][m]

                        error_dict[m] = sign_error(c1, c2, g1, g2)

            for epoch in run1.epoch_measures:
                if epoch not in run2.epoch_measures:
                    continue

                error_dict = getattr(self, "epochs_{}_error".format(st))[epoch] = {}

                for m in run1.epoch_measures[epoch]:
                    c1 = run1.epoch_measures[epoch][m]
                    c2 = run2.epoch_measures[epoch][m]

                    error_dict[m] = sign_error(c1, c2, g1, g2)


class PreComp:
    def __init__(self, runs):
        self.runs = {x[0]: x[1] for x in take(50, runs.items())}
        self.run_pairs = {}

        for hps_1 in tqdm(self.runs):
            for hps_2 in self.runs:
                if hps_2 < hps_1:
                    continue  # don't double count weights

                key = (hps_1, hps_2)
                self.run_pairs[key] = {}

                runs_1 = self.runs[hps_1]
                runs_2 = self.runs[hps_2]

                for r1 in runs_1:
                    for r2 in runs_2:
                        self.run_pairs[key][(r1.seed, r2.seed)] = RunPair(r1, r2)


if __name__ == "__main__":
    with open("./data/new/runs_all.pickle", "rb") as f:
        runs = pickle.load(f)

    pre = PreComp(runs)

    with open("./data/pre-comp/pre-comp_all.pickle", "wb+") as f:
        pickle.dump(pre, f, protocol=pickle.HIGHEST_PROTOCOL)
