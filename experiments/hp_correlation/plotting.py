import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import os
import pickle
from correlations import overall_correlation


def overall_corr_figs():
    steps = [0.001, 0.01, 0.1, 1] + list(range(5, 301, 5))
    corrs = {}
    corrs["test_acc"] = []

    for a in ["margin", "exponential_margin", "normalized_exponential_margin", "shifted_margin",
              "pac_bayes", "vc_dim", "norms", "optimisation"]:
        with open("./data.nosync/pre-comp-correlations-future/{}-best.pickle".format(a), "rb") as f:
            pair_corrs = pickle.load(f)

        corrs[a] = []
        overall_corrs = overall_correlation(pair_corrs)

        for s in steps:
            max_corr = 0
            for x in overall_corrs["weighted"][s]:
                c = abs(np.mean([b[0] for b in overall_corrs["weighted"][s][x] if b[2] > 1]))

                if x == "accuracy/test":
                    corrs["test_acc"].append(c)
                elif c > max_corr:
                    max_corr = c

            corrs[a].append(max_corr)

    margin = []
    for i in range(len(steps)):
        margin.append(max(corrs["margin"][i], corrs["exponential_margin"][i],
                          corrs["normalized_exponential_margin"][i], corrs["shifted_margin"][i]))

    corrs_new = {k: v for k, v in corrs.items() if k in ["pac_bayes", "vc_dim", "norms", "optimisation", "test_acc"]}
    corrs_new["margin"] = margin

    plt.figure(figsize=(10, 7))
    for x in corrs_new:
        plt.plot(steps[:5], corrs_new[x][:5], label=x)

    plt.ylim([0, 0.7])
    plt.xlabel("epoch")
    plt.ylabel("absolute correlation")
    plt.legend()
    plt.show()


def correlation_envelope(correlations, measure, steps):
    x = steps
    # mean
    y = [np.mean([x[0] for x in correlations[s][measure]]) for s in steps]
    error = [np.std([x[0] for x in correlations[s][measure]]) for s in steps]
    # median
    # y = [np.median([x[0] for x in correlations[s][measure]]) for s in steps]
    # upper = [np.percentile([x[0] for x in correlations[s][measure]], 25) for s in steps]
    # lower = [np.percentile([x[0] for x in correlations[s][measure]], 75) for s in steps]

    plt.plot(x, y, "k-")
    plt.title(measure)
    # mean
    # plt.fill_between(x, [max(y[i]-error[i], -1) for i in range(len(y))], [min(y[i]+error[i], 1) for i in range(len(y))])
    # median
    # plt.fill_between(x, lower, upper)
    plt.show()
