import pickle
import numpy as np
from save_measures import Run, ExperimentType
from tqdm import tqdm
from config import measure_types, stop_types
from math import ceil


def sign_error(c1, c2, g1, g2):
    sign = np.sign(c1 - c2) * np.sign(g1 - g2)
    return 0 if np.isnan(sign) else sign


def hoeffding_weight(delta_gen, m=10000):
    """
    This value has the following guarantee. If your measurement of the generalization gap is computed
    using m (say m=10,000) independent samples, then accepting samples only when this value > p would
    mean that the difference of the two generalization estimates has the correct sign with probability >= p
    """
    phi = 2 * np.exp(-2 * m * (np.abs(delta_gen) / 2) ** 2)

    return max(0., 1. - phi) ** 2


class RunPair:
    def __init__(self, run1, run2, measure_type, stop_type):
        self.stop_type = stop_type
        self.measure_type = measure_type
        self.weight = hoeffding_weight(getattr(run1, "{}_test_acc".format(self.stop_type)) -
                                       getattr(run2, "{}_test_acc".format(self.stop_type)))
        self.errors = {}

        if not (stop_type == "_99" and (run1._99_step[0] is np.inf or run2._99_step[0] is np.inf)):
            self.setup_errors(run1, run2)

    def setup_errors(self, run1, run2):
        min_max = min(max(run1.measures), max(run2.measures))
        max_max = max(max(run1.measures), max(run2.measures))
        max_run = run1 if max(run1.measures) >= max(run2.measures) else run2
        min_run = run2 if max_run == run1 else run1

        g_max = getattr(max_run, "{}_test_acc".format(self.stop_type))
        g_min = getattr(min_run, "{}_test_acc".format(self.stop_type))

        for step in max_run.measures:
            if step > 300:
                continue

            if step in min_run.measures:
                error_dict = self.errors[step] = {}

                for m in measure_types[self.measure_type]:
                    try:
                        c_max = max_run.measures[step][m]
                        c_min = min_run.measures[step][m]
                    except KeyError:
                        continue

                    error_dict[m] = sign_error(c_max, c_min, g_max, g_min)

            elif step > min_max:
                error_dict = self.errors[step] = {}

                for m in measure_types[self.measure_type]:
                    try:
                        c_max = max_run.measures[step][m]
                        c_min = min_run.measures[min_max][m]
                    except KeyError:
                        continue
                    error_dict[m] = sign_error(c_max, c_min, g_max, g_min)

    # def setup_errors(self, run1, run2):
    #     g1 = getattr(run1, "{}_test_acc".format(self.stop_type))
    #     g2 = getattr(run2, "{}_test_acc".format(self.stop_type))
    #
    #     for step in run1.measures:
    #         if step not in run2.measures:
    #             continue
    #
    #         error_dict = self.errors[step] = {}
    #
    #         for m in measure_types[self.measure_type]:
    #             try:
    #                 c1 = run1.measures[step][m]
    #                 c2 = run2.measures[step][m]
    #             except KeyError:
    #                 continue
    #
    #             error_dict[m] = sign_error(c1, c2, g1, g2)


def make_pairs(runs, measure_type, stop_type):
    run_pairs = {}

    for hps_1 in tqdm(runs):
        for hps_2 in runs:
            if hps_2 < hps_1:
                continue  # don't double count weights

            key = (hps_1, hps_2)
            run_pairs[key] = {}

            runs_1 = runs[hps_1]
            runs_2 = runs[hps_2]

            for r1 in runs_1:
                for r2 in runs_2:
                    run_pairs[key][(r1.seed, r2.seed)] = RunPair(r1, r2, measure_type, stop_type)

    return run_pairs


if __name__ == "__main__":
    with open("./data.nosync/runs/all.pickle", "rb") as f:
        runs = pickle.load(f)

    for stop in stop_types:
        for meas in measure_types:
            pairs = make_pairs(runs, meas, stop)

            with open("./data.nosync/pre-comp-pairs-future/{}-{}.pickle".format(meas, stop), "wb+") as f:
                pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

            del pairs
