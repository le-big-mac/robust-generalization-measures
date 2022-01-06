from precomputation import RunPair
from config import params, ExperimentType


def differ_in_only(config_pair, hp):
    i = params.index(hp)

    truth = True
    for j in range(len(params)):
        if j == i:
            truth = truth and config_pair[0][0][j] != config_pair[1][0][j]
        else:
            truth = truth and config_pair[0][0][j] == config_pair[1][0][j]

    return truth


def conditional_correlation(pairs, condition, weighted=True):
    corrs = {}

    for hp_pair, run_pair in pairs.items():
        if not condition(hp_pair):
            continue

        for step, measures in run_pair.corrs.items():
            step_corrs = corrs.get(step, {})

            for measure, corr in measures.items():
                step_measure_corrs = step_corrs.get(measure, [])
                if weighted:
                    step_measure_corrs.append(corr * run_pair.weight)
                else:
                    step_measure_corrs.append(corr)
                step_corrs[measure] = step_measure_corrs

            corrs[step] = step_corrs

    return corrs


def hp_correlation(pair_corrs, hp, weighted=True):
    def condition(hp_pair):
        return differ_in_only(hp_pair, hp) and \
               ((hp == "batch_norm" and ExperimentType.BATCH_NORM in hp_pair[0][1] and
                 ExperimentType.BATCH_NORM in hp_pair[1][1]) or
                (hp != "batch_norm" and ExperimentType.BATCH_NORM not in hp_pair[0][1] and
                 ExperimentType.BATCH_NORM not in hp_pair[0][1]))

    return conditional_correlation(pair_corrs, condition, weighted=weighted)


def initialization_correlation(pair_corrs, weighted=True):
    def condition(hp_pair):
        return hp_pair[0] == hp_pair[1]

    return conditional_correlation(pair_corrs, condition, weighted=weighted)


def overall_correlation(pair_corrs, weighted=True):
    def condition(hp_pair):
        return ExperimentType.BATCH_NORM not in hp_pair[0][1] and ExperimentType.BATCH_NORM not in hp_pair[1][1]

    return conditional_correlation(pair_corrs, condition, weighted=weighted)
