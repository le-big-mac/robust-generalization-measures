import pickle
import numpy as np
from tqdm import tqdm
from config import measure_types, ExperimentType
from math import ceil


def sign_correlation(c1, c2, g1, g2):
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
        self.corrs = {}

        if not (stop_type == "_99" and (run1._99_step[0] is np.inf or run2._99_step[0] is np.inf)):
            self.setup_corrs(run1, run2)

    def setup_corrs(self, run1, run2):
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
                corr_dict = self.corrs[step] = {}

                for m in measure_types[self.measure_type]:
                    try:
                        c_max = max_run.measures[step][m]
                        c_min = min_run.measures[step][m]
                    except KeyError:
                        continue

                    corr_dict[m] = sign_correlation(c_max, c_min, g_max, g_min)

            elif step > min_max:
                corr_dict = self.corrs[step] = {}

                for m in measure_types[self.measure_type]:
                    try:
                        c_max = max_run.measures[step][m]
                        c_min = min_run.measures[min_max][m]
                    except KeyError:
                        continue
                    corr_dict[m] = sign_correlation(c_max, c_min, g_max, g_min)

        for step in range(5 * ceil((max_max+1)/5), 301, 5):
            self.corrs[step] = self.corrs[max_max]


def make_pairs(runs, measure_type, stop_type):
    run_pairs = {}

    for hps_1 in tqdm(runs):
        for hps_2 in runs:
            if hps_2 < hps_1 or hps_2 == hps_1:
                continue  # don't double count weights

            key = (hps_1, hps_2)
            run_pairs[key] = {}

            r1 = runs[hps_1]
            r2 = runs[hps_2]

            run_pairs[key] = RunPair(r1, r2, measure_type, stop_type)

    return run_pairs


if __name__ == "__main__":
    for seed in [0, 17, 43]:
        with open("./new_data.nosync/runs/all-{}.pickle".format(seed), "rb") as f:
            runs = pickle.load(f)

        for meas in measure_types:
            pairs = make_pairs(runs, meas, "best")

            with open("./new_data.nosync/pre-comp/{}-{}-{}.pickle".format(meas, "best", seed), "wb+") as f:
                pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

            del pairs
