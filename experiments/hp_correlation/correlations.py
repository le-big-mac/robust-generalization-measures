import pickle
import marshal
from config import stop_types, measure_types
from tqdm import tqdm
from precomputation import RunPair
from config import params, ExperimentType
from math import ceil


def differ_in_only(config_pair, hp):
    i = params.index(hp)

    truth = True
    for j in range(len(params)):
        if j == i:
            continue
        truth = truth and config_pair[0][0][j] == config_pair[1][0][j]

    return truth


def conditional_correlation(pair_corrs, condition):
    def extract_corrs(name):
        corr_dict = pair_corrs[name]
        corrs = {}

        for hp_pair, steps_dict in corr_dict.items():
            if not condition(hp_pair):
                continue

            for step, measures in steps_dict.items():
                if name == "weighted":
                    measures_dict, num_pairs, _ = measures
                else:
                    measures_dict, num_pairs = measures

                step_corrs = corrs.get(step, {})

                for measure, corr in measures_dict.items():
                    step_measure_corrs = step_corrs.get(measure, [])
                    step_measure_corrs.append((corr, num_pairs))
                    step_corrs[measure] = step_measure_corrs

                corrs[step] = step_corrs

        return corrs

    weighted_corrs = extract_corrs("weighted")
    unweighted_corrs = extract_corrs("unweighted")

    return {"weighted": weighted_corrs, "unweighted": unweighted_corrs}


def hp_correlation(pair_corrs, hp):
    def condition(hp_pair):
        return differ_in_only(hp_pair, hp) and \
               ((hp == "batch_norm" and hp_pair[0][1] == ExperimentType.BATCH_NORM and
                 hp_pair[1][1] == ExperimentType.BATCH_NORM) or
                (hp != "batch_norm" and hp_pair[0][1] != ExperimentType.BATCH_NORM and
                 hp_pair[1][1] != ExperimentType.BATCH_NORM))

    return conditional_correlation(pair_corrs, condition)


def initialization_correlation(pair_corrs):
    def condition(hp_pair):
        return hp_pair[0] == hp_pair[1] and hp_pair[0][1] != ExperimentType.BATCH_NORM and \
               hp_pair[1][1] != ExperimentType.BATCH_NORM

    return conditional_correlation(pair_corrs, condition)


def overall_correlation(pair_corrs):
    def condition(hp_pair):
        return hp_pair[0][1] != ExperimentType.BATCH_NORM and hp_pair[1][1] != ExperimentType.BATCH_NORM

    return conditional_correlation(pair_corrs, condition)


def precompute_pair_correlations(pairs):
    weighted_corrs = {}  # store in the format (sum of sign error, number of runs, total weight)
    unweighted_corrs = {}

    for hp_pair, seed_pair_dict in tqdm(pairs.items()):
        try:
            weighted_corrs[hp_pair], unweighted_corrs[hp_pair] = \
                pair_average_sign_error(seed_pair_dict, same_hps=(hp_pair[0] == hp_pair[1]))
        except ValueError:
            print(hp_pair)
            raise ValueError

    return {"weighted": weighted_corrs, "unweighted": unweighted_corrs}


def pair_average_sign_error(run_pair_dict, same_hps=False):
    collect_errors = {}

    for seeds, run_pair in run_pair_dict.items():
        if same_hps and seeds[0] >= seeds[1]:
            continue

        weight = run_pair.weight
        max_step = max(run_pair.errors)
        round_max_step = 5*(ceil((max_step+1)/5))

        for step, measure_sign_error in run_pair.errors.items():
            weighted_errors, unweighted_errors, total_weight, weighted_pairs, total_pairs = \
                collect_errors.get(step, ({}, {}, 0, 0, 0))

            for measure, e in measure_sign_error.items():
                weighted_errors[measure] = weighted_errors.get(measure, 0) + weight * e
                unweighted_errors[measure] = unweighted_errors.get(measure, 0) + e

            collect_errors[step] = (weighted_errors, unweighted_errors,
                                    total_weight + weight, weighted_pairs + int(weight > 0), total_pairs + 1)

        for step in range(round_max_step, 301, 5):
            measure_sign_error = run_pair.errors[max_step]

            weighted_errors, unweighted_errors, total_weight, weighted_pairs, total_pairs = \
                collect_errors.get(step, ({}, {}, 0, 0, 0))

            for measure, e in measure_sign_error.items():
                weighted_errors[measure] = weighted_errors.get(measure, 0) + weight * e
                unweighted_errors[measure] = unweighted_errors.get(measure, 0) + e

            collect_errors[step] = (weighted_errors, unweighted_errors,
                                    total_weight + weight, weighted_pairs + int(weight > 0), total_pairs + 1)

    weighted = {}
    unweighted = {}
    for step, collected in collect_errors.items():
        weighted_errors, unweighted_errors, total_weight, weighted_pairs, num_pairs = collected

        if total_weight == 0:
            weighted[step] = ({}, 0, 0)
        else:
            weighted[step] = ({x[0]: x[1]/total_weight for x in weighted_errors.items()}, weighted_pairs, total_weight)

        unweighted[step] = ({x[0]: x[1]/num_pairs for x in unweighted_errors.items()}, num_pairs)

        any_range = any(abs(x) > 1 for x in weighted[step][0].values())
        if any_range:
            print("{}: {}".format(step, any_range))
            print({(k, v) for k, v in weighted[step][0].items() if abs(v) > 1})
            raise ValueError

    return weighted, unweighted


if __name__ == "__main__":
    for st in stop_types:
        for m in measure_types:
            with open("./data.nosync/pre-comp-pairs-future/{}-{}.pickle".format(m, st), "rb") as f:
                pairs = pickle.load(f)

            pair_corrs = precompute_pair_correlations(pairs)

            with open("./data.nosync/pre-comp-correlations-future/{}-{}.pickle".format(m, st), "wb+") as f:
                pickle.dump(pair_corrs, f)

            del pair_corrs
