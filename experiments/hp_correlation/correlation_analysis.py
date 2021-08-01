import pandas as pd
import scipy.stats as stats
import numpy as np
import pickle
from collections import defaultdict
import sys
import os
from save_hp_measures import save_hp_measures

results_order = ["PARAMS", "INVERSE_MARGIN", "LOG_SPEC_INIT_MAIN", "LOG_SPEC_INIT_MAIN_FFT", "LOG_SPEC_ORIG_MAIN",
                 "LOG_SPEC_ORIG_MAIN_FFT", "LOG_PROD_OF_SPEC_OVER_MARGIN", "LOG_SUM_OF_SPEC_OVER_MARGIN_FFT",
                 "LOG_PROD_OF_SPEC", "LOG_PROD_OF_SPEC_FFT", "LOG_SUM_OF_SPEC_OVER_MARGIN",
                 "LOG_SUM_OF_SPEC_OVER_MARGIN_FFT", "LOG_SUM_OF_SPEC", "LOG_SUM_OF_SPEC_FFT", "DIST_SPEC_INIT",
                 "DIST_SPEC_INIT_FFT", "FRO_OVER_SPEC", "FRO_OVER_SPEC_FFT", "LOG_PROD_OF_FRO_OVER_MARGIN",
                 "LOG_PROD_OF_FRO", "LOG_SUM_OF_FRO_OVER_MARGIN", "LOG_SUM_OF_FRO", "FRO_DIST", "PARAM_NORM",
                 "PATH_NORM_OVER_MARGIN", "PATH_NORM", "PACBAYES_INIT", "PACBAYES_ORIG", "PACBAYES_FLATNESS",
                 "PACBAYES_MAG_INIT", "PACBAYES_MAG_ORIG", "PACBAYES_MAG_FLATNESS", "SOTL", "SOTL_10", "L2", "L2_DIST",
                 "cross_entropy_epoch_end/train", "accuracy/train", "VALIDATION_ACC"]


def dd_list():
    return defaultdict(list)


def kendall_correlations(hp: str, fill_nan_epochs: bool = True, filter_train_acc: float = 0):
    if os.path.isfile("./results/{}/gen_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, fill_nan_epochs, filter_train_acc)) and \
            os.path.isfile("./results/{}/acc_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
                hp, fill_nan_epochs, filter_train_acc)):
        return

    if not os.path.isfile("./results/{}/measures.pickle".format(hp)):
        save_hp_measures(hp)

    with open("./results/{}/measures.pickle".format(hp), "rb") as f:
        key_measures_dict = pickle.load(f)

    gen_correlation_dict = defaultdict(dd_list)
    acc_correlation_dict = defaultdict(dd_list)

    for fixed_params, epoch_measure_dict in key_measures_dict.items():
        # if final train accuracy of any run in group is below filter accuracy remove it
        if (epoch_measure_dict[sys.maxsize]["acc"] + epoch_measure_dict[sys.maxsize]["gen"] < filter_train_acc).any():
            continue

        for epoch, epoch_measures in epoch_measure_dict.items():
            # if fill_nan_epochs is true then fill any nan rows with the values from epoch == sys.maxsize
            if not fill_nan_epochs and epoch_measures.isnull().values.any():
                continue
            elif epoch_measures.isnull().values.any():
                for index, row in epoch_measures.iterrows():
                    if row.isnull().values.any():
                        epoch_measures.loc[index] = epoch_measure_dict[sys.maxsize].loc[index]

            epoch_measures = epoch_measures.astype(np.float64)

            for meas in epoch_measures:
                meas_str = meas[11:] if meas[:11] == "complexity/" else meas

                gen_corr, _ = stats.kendalltau(epoch_measures[meas], epoch_measures["gen"])
                gen_correlation_dict[meas_str][epoch].append(gen_corr)
                acc_corr, _ = stats.kendalltau(epoch_measures[meas], epoch_measures["acc"])
                acc_correlation_dict[meas_str][epoch].append(acc_corr)

    with open("./results/{}/gen_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, fill_nan_epochs, filter_train_acc), "wb+") as f:
        pickle.dump(gen_correlation_dict, f)
    with open("./results/{}/acc_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, fill_nan_epochs, filter_train_acc), "wb+") as g:
        pickle.dump(acc_correlation_dict, g)


def load_correlation_dict(hp: str, corr_type: str, fill_nan_epochs: bool = True, filter_train_acc: float = 0):
    if not os.path.isfile("./results/{}/{}_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, corr_type, fill_nan_epochs, filter_train_acc)):
        kendall_correlations(hp, fill_nan_epochs, filter_train_acc)

    with open("./results/{}/{}_correlation-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, corr_type, fill_nan_epochs, filter_train_acc), "rb") as f:
        correlation_dict = pickle.load(f)

    return correlation_dict


def make_stats_csv(hp: str, corr_type: str, min_corrs_epoch: int, fill_nan_epochs: bool = True,
                   filter_train_acc: float = 0):
    correlation_dict = load_correlation_dict(hp, corr_type, fill_nan_epochs, filter_train_acc)

    csv_order = results_order[:]
    csv_order.append(hp)

    min_corrs_epoch = len(correlation_dict[hp][1]) if min_corrs_epoch == 0 else min_corrs_epoch

    with open("./results/{}/{}_stats-min_corrs_{}-fill_nan_epochs_{}-min_acc_{}-figs.csv".format(
            hp, corr_type, min_corrs_epoch, fill_nan_epochs, filter_train_acc), "w+") as csv_file:
        for k in csv_order:
            measure_epoch_corrs = correlation_dict[k]
            csv_str = "{},".format(k)

            for epoch, corrs in sorted(measure_epoch_corrs.items()):
                # only include epochs for which we have correlation values for at least min_corrs_epochs families
                if len(corrs) < min_corrs_epoch:
                    continue
                csv_str += "{},".format(epoch)
                csv_str += make_stats_string(corrs)

            csv_file.write(csv_str)


def make_stats_string(corrs):
    mean = sum(corrs) / len(corrs)
    csv_str = "{},".format(mean)
    worst = min(corrs) if mean >= 0 else max(corrs)
    csv_str += "{},".format(worst)
    csv_str += "{},".format(np.var(corrs))
    csv_str += "{},".format(np.median(corrs))
    csv_str += "{},".format(np.quantile(corrs, 0.25))
    csv_str += "{},".format(np.quantile(corrs, 0.75))
    csv_str += "\n"

    return csv_str


kendall_correlations("model_depth", True, 0.99)
