import pickle
from math import sqrt
import numpy as np
from save_measures import Run

with open("./experiments/hp_correlation/data/old/runs.pickle", "rb") as f:
    runs = pickle.load(f)

inv_margins = []
for k in runs:
    for r in runs[k]:
        for step in r.step_measures:
            if r.step_measures[step]["accuracy/train"] < 0.9:
                inv_margins.append(- sqrt(50000) * r.step_measures[step]["INVERSE_MARGIN"])
            else:
                inv_margins.append(sqrt(50000) * r.step_measures[step]["INVERSE_MARGIN"])

neg_margins = [1/x for x in inv_margins if x < 0]
