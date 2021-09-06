import scipy.stats as stats
import numpy as np
import pickle
from collections import defaultdict
import os
from save_measures import save_hp_measures
from typing import List
import math

results_order = ["PARAMS", "INVERSE_MARGIN", "LOG_SPEC_INIT_MAIN_FFT", "LOG_SPEC_ORIG_MAIN_FFT",
                 "LOG_SUM_OF_SPEC_OVER_MARGIN_FFT", "LOG_SUM_OF_SPEC_FFT", "LOG_PROD_OF_SPEC_OVER_MARGIN_FFT",
                 "LOG_PROD_OF_SPEC_FFT", "DIST_SPEC_INIT_FFT", "FRO_OVER_SPEC_FFT", "LOG_PROD_OF_FRO_OVER_MARGIN",
                 "LOG_PROD_OF_FRO", "LOG_SUM_OF_FRO_OVER_MARGIN", "LOG_SUM_OF_FRO", "FRO_DIST", "PARAM_NORM",
                 "PATH_NORM_OVER_MARGIN", "PATH_NORM", "PACBAYES_INIT", "PACBAYES_ORIG", "PACBAYES_FLATNESS",
                 "PACBAYES_MAG_INIT", "PACBAYES_MAG_ORIG", "PACBAYES_MAG_FLATNESS", "SOTL", "SOTL_10", "L2", "L2_DIST",
                 "cross_entropy_epoch_end/train", "accuracy/train", "VALIDATION_ACC", "hp"]


def dd_list():
    return defaultdict(list)


def overall_correlation(run_list: List, epochs: List[int], stop_type: str, name: str, filter_train_acc: float = 0.99):
    epoch_measures_dict = defaultdict(dd_list)
    for param_dict, run_measures in run_list:
        if run_measures.loc["final"]["accuracy/train"] < filter_train_acc or \
                math.isnan(stop_type == "99" and run_measures.loc["final"]["99_test_acc"]):
            continue

        for meas in run_measures:
            meas_str = meas[11:] if meas[:11] == "complexity/" else meas
            if meas_str not in results_order and \
                    meas_str not in ["{}_gen_error".format(stop_type), "{}_test_acc".format(stop_type)]:
                continue

            for e in epochs:
                try:
                    measure = run_measures.loc[e][meas]
                except KeyError:
                    measure = run_measures.loc["final"][meas]
                epoch_measures_dict[e][meas_str].append(measure)

            epoch_measures_dict["final"][meas_str].append(run_measures.loc["final"][meas])

    print(epoch_measures_dict.keys())
    gen_correlation_dict = defaultdict(dd_list)
    acc_correlation_dict = defaultdict(dd_list)
    for epoch, measures_dict in epoch_measures_dict.items():
        print("Epoch: {}".format(epoch))
        for meas, values in measures_dict.items():
            print("Meas: {}".format(meas))
            gen_corr, _ = stats.kendalltau(values, measures_dict["{}_gen_error".format(stop_type)])
            print("Gen corr: {}".format(gen_corr))
            gen_correlation_dict[meas][epoch].append(gen_corr)
            acc_corr, _ = stats.kendalltau(values, measures_dict["{}_test_acc".format(stop_type)])
            print("Acc corr: {}".format(acc_corr))
            acc_correlation_dict[meas][epoch].append(acc_corr)
            print()

    with open("./results/overall/gen_correlation-{}-type_{}-min_acc_{}.pickle".format(
            name, stop_type, filter_train_acc), "wb+") as f:
        pickle.dump(gen_correlation_dict, f)
    with open("./results/overall/acc_correlation-{}-type_{}-min_acc_{}.pickle".format(
            name, stop_type, filter_train_acc), "wb+") as g:
        pickle.dump(acc_correlation_dict, g)


def overall_correlation_csv(file_name: str):
    with open("./results/overall/{}.pickle".format(file_name), "rb") as f:
        correlation_dict = pickle.load(f)

    csv_str = ""
    for measure in results_order:
        epoch_dict = correlation_dict[measure]
        csv_str += "{},".format(measure)

        for epoch in sorted(epoch_dict.keys(), key=lambda v: (isinstance(v, str), v)):
            corr = epoch_dict[epoch][0]
            csv_str += "{},{},".format(epoch, corr)

        csv_str += "\n"

    with open("./results/overall/{}.csv".format(file_name), "w+") as f:
        f.write(csv_str)


def hp_kendall_correlations(hp: str, epochs: List[int], type: str, fill_nan_epochs: bool = True,
                            filter_train_acc: float = 0.99):
    if os.path.isfile("./results/{}/gen_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, type, fill_nan_epochs, filter_train_acc)) and \
            os.path.isfile("./results/{}/acc_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
                hp, type, fill_nan_epochs, filter_train_acc)):
        return

    if not os.path.isfile("./results/{}/measures_{}.pickle".format(hp, epochs)):
        save_hp_measures(hp, epochs)

    with open("./results/{}/measures_{}.pickle".format(hp, epochs), "rb") as f:
        key_measures_dict = pickle.load(f)

    gen_correlation_dict = defaultdict(dd_list)
    acc_correlation_dict = defaultdict(dd_list)

    for fixed_params, epoch_measure_dict in key_measures_dict.items():
        # if final train accuracy of any run in group is below filter accuracy then filter out entire group
        if (epoch_measure_dict["final"]["accuracy/train"] < filter_train_acc).any() or \
                (type == "99" and epoch_measure_dict["final"]["99_test_acc"].isna().any()):
            continue

        for epoch, epoch_measures in epoch_measure_dict.items():
            # if fill_nan_epochs is true then fill any nan rows (model had converged before this epoch)
            # with the values from the final epoch
            if not fill_nan_epochs and epoch_measures.isnull().values.any():
                continue
            elif epoch_measures.isnull().values.any():
                for index, row in epoch_measures.iterrows():
                    if row.isnull().values.any():
                        epoch_measures.loc[index] = epoch_measure_dict["final"].loc[index]

            epoch_measures = epoch_measures.astype(np.float64)

            # if not (epoch_measures["complexity/SOTL"] == epoch_measures["complexity/SOTL_10"]).all() and epoch != "final" \
            #         and epoch <= 10:
            #     print(fixed_params)
            #     print(epoch)
            #     print(epoch_measures["complexity/SOTL"])
            #     print(epoch_measures["complexity/SOTL_10"])

            for meas in epoch_measures:
                meas_str = meas[11:] if meas[:11] == "complexity/" else meas

                gen_corr, _ = stats.kendalltau(epoch_measures[meas], epoch_measures["{}_gen_error".format(type)])
                gen_correlation_dict[meas_str][epoch].append(gen_corr)
                acc_corr, _ = stats.kendalltau(epoch_measures[meas], epoch_measures["{}_test_acc".format(type)])
                acc_correlation_dict[meas_str][epoch].append(acc_corr)

            gen_corr_hp, _ = stats.kendalltau(epoch_measures.index, epoch_measures["{}_gen_error".format(type)])
            gen_correlation_dict["hp"][epoch].append(gen_corr_hp)
            acc_corr_hp, _ = stats.kendalltau(epoch_measures.index, epoch_measures["{}_test_acc".format(type)])
            acc_correlation_dict["hp"][epoch].append(acc_corr_hp)

    with open("./results/{}/gen_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, type, fill_nan_epochs, filter_train_acc), "wb+") as f:
        pickle.dump(gen_correlation_dict, f)
    with open("./results/{}/acc_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, type, fill_nan_epochs, filter_train_acc), "wb+") as g:
        pickle.dump(acc_correlation_dict, g)


# Overall measure used in Fantastic Generalization Measures
def average_hp_correlations(hps: List[str], corr_type: str, epochs: List[int], stop_type: str,
                            fill_nan_epochs: bool = True, filter_train_acc: float = 0.99):
    epochs.append("final")

    measure_epoch_average_corrs = {}
    for hp in hps:
        correlation_dict = load_correlation_dict(hp, epochs, corr_type, stop_type, fill_nan_epochs, filter_train_acc)

        for measure in results_order:
            epoch_corrs = correlation_dict[measure]

            if measure not in measure_epoch_average_corrs:
                measure_epoch_average_corrs[measure] = {}

            for e in epochs:
                if e not in measure_epoch_average_corrs[measure]:
                    measure_epoch_average_corrs[measure][e] = {}

                corrs = epoch_corrs[e].copy()
                corrs = [0 if math.isnan(x) else x for x in corrs]
                print("Epoch {} num corrs {}".format(e, len(corrs)))
                if len(corrs) == 0:
                    av = 0
                else:
                    av = sum(corrs)/len(corrs)

                measure_epoch_average_corrs[measure][e][hp] = av

    for measure, epoch_hp_av_corrs in measure_epoch_average_corrs.items():
        print("Measure: {}".format(measure))
        for epoch, hp_av_corrs in epoch_hp_av_corrs.items():
            hp_av_corrs["average"] = sum(hp_av_corrs.values())/len(hp_av_corrs)
            print("Epoch: {}".format(epoch))
            print("Av corrs: {}".format(hp_av_corrs))
            print("Av corr over hps: {}".format(hp_av_corrs["average"]))

    hp_str = "_".join(hps)
    with open("./results/all/av_{}_correlation_over_{}-type_{}-fill_nan_epochs-{}-min_acc_{}.pickle".format(
            corr_type, hp_str, stop_type, fill_nan_epochs, filter_train_acc), "wb+") as f:
        pickle.dump(measure_epoch_average_corrs, f)

    with open("./results/all/av_{}_correlation_over_{}-type_{}-fill_nan_epochs-{}-min_acc_{}.csv".format(
            corr_type, hp_str, stop_type, fill_nan_epochs, filter_train_acc), "w+") as f:

        for measure, epoch_corrs in measure_epoch_average_corrs.items():
            csv_str = "{},".format(measure)

            for epoch in epochs:
                csv_str += "{},".format(epoch)
                csv_str += "{},".format(epoch_corrs[epoch]["average"])

            csv_str += "\n"
            f.write(csv_str)


def load_correlation_dict(hp: str, epochs: List[int], corr_type: str, stop_type: str, fill_nan_epochs: bool = True,
                          filter_train_acc: float = 0):
    if not os.path.isfile("./results/{}/{}_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, corr_type, stop_type, fill_nan_epochs, filter_train_acc)):
        hp_kendall_correlations(hp, epochs, stop_type, fill_nan_epochs, filter_train_acc)

    with open("./results/{}/{}_correlation-type_{}-fill_nan_epochs_{}-min_acc_{}.pickle".format(
            hp, corr_type, stop_type, fill_nan_epochs, filter_train_acc), "rb") as f:
        correlation_dict = pickle.load(f)

    return correlation_dict


def make_stats_csv(hp: str, epochs: List[int], corr_type: str, stop_type: str, min_corrs_epoch: int,
                   fill_nan_epochs: bool = True, filter_train_acc: float = 0):
    correlation_dict = load_correlation_dict(hp, epochs, corr_type, stop_type, fill_nan_epochs, filter_train_acc)

    min_corrs_epoch = len(correlation_dict["hp"]["final"]) if min_corrs_epoch == 0 else min_corrs_epoch

    with open("./results/{}/{}_stats-type_{}-min_corrs_{}-fill_nan_epochs_{}-min_acc_{}.csv".format(
            hp, corr_type, stop_type, min_corrs_epoch, fill_nan_epochs, filter_train_acc), "w+") as csv_file:
        for meas in results_order:
            measure_epoch_corrs = correlation_dict[meas]
            csv_str = "{},".format(meas)

            print(meas)
            print(measure_epoch_corrs.items())
            for epoch in epochs:
                if epoch not in measure_epoch_corrs:
                    continue
                corrs = measure_epoch_corrs[epoch]
                if len(corrs) < min_corrs_epoch:
                    continue
                csv_str += "{},".format(epoch)
                csv_str += make_stats_string(corrs)

            csv_str += "final,"
            csv_str += make_stats_string(measure_epoch_corrs["final"])
            csv_str += "\n"

            csv_file.write(csv_str)


def make_stats_string(corrs):
    corrs = corrs.copy()
    corrs = [0 if math.isnan(x) else x for x in corrs]
    if len(corrs) == 0:
        csv_str = "nan,nan,nan,nan,nan,nan,"
    else:
        mean = sum(corrs) / len(corrs)
        csv_str = "{},".format(mean)
        worst = min(corrs) if mean >= 0 else max(corrs)
        csv_str += "{},".format(worst)
        csv_str += "{},".format(np.var(corrs))
        csv_str += "{},".format(np.median(corrs))
        csv_str += "{},".format(np.quantile(corrs, 0.25))
        csv_str += "{},".format(np.quantile(corrs, 0.75))

    return csv_str


if __name__ == "__main__":
    for name in ["basic", "batch_norm", "dropout", "restricted_lr"]:
        with open("./data/runs_{}_[1, 5, 10, 15, 20].pickle".format(name), "rb") as f:
            run_list = pickle.load(f)
        a = [("final", 0), ("best", 0), ("99", 0), ("final", 0.99), ("best", 0.99)]
        for stop_type, filter_acc in a:
            overall_correlation(run_list, [1, 5, 10, 15, 20], stop_type, name, filter_acc)
            overall_correlation_csv("acc_correlation-{}-type_{}-min_acc_{}".format(name, stop_type, filter_acc))
