import pickle
import numpy as np

stop_types = ["final", "best", "_99"]
early_batches = (0, 1, 10, 100)


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
        self.r1 = run1
        self.r2 = run2

        for st in stop_types:
            setattr(self, "{}_weight".format(st), hoeffding_weight(getattr(self.r1, "{}_test_acc".format(st)) -
                                                                   getattr(self.r2, "{}_test_acc".format(st))))
            setattr(self, "early_batches_{}_error".format(st), {})
            setattr(self, "epochs_{}_error".format(st), {})

        complexity_measures = self.r1.step_measures[0].keys()

        for st in stop_types:
            g1 = getattr(self.r1, "{}_test_acc".format(st))
            g2 = getattr(self.r2, "{}_test_acc".format(st))

            for step in early_batches:
                error_dict = getattr(self, "early_batches_{}_error".format(st))[step] = {}

                for m in complexity_measures:
                    c1 = self.r1.step_measures[step][m]
                    c2 = self.r1.step_measures[step][m]

                    error_dict[m] = sign_error(c1, c2, g1, g2)

            for epoch in self.r1.epoch_measures:
                if epoch not in self.r2.epoch_measures:
                    continue

                error_dict = getattr(self, "epochs_{}_error".format(st))[epoch] = {}

                for m in complexity_measures:
                    c1 = self.r1.epoch_measures[epoch][m]
                    c2 = self.r2.epoch_measures[epoch][m]

                    error_dict[m] = sign_error(c1, c2, g1, g2)


class PreComp:
    def __init__(self, runs):
        self.runs = runs
        self.run_pairs = {}

        for hps_1 in self.runs:
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
    pass
