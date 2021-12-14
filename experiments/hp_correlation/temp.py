import pickle
from math import sqrt
import sys
import numpy as np
from save_measures import Run


def calculate_margins():
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


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
