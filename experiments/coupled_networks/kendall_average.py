import wandb
import pandas as pd
import scipy.stats as stats
from collections import defaultdict

api = wandb.Api()
runs = api.runs("rgm")

gens = []
frames = []
hparams = ['lr', 'seed', 'data_seed', 'batch_norm', 'batch_size', 'model_depth', 'dropout_prob', 'weight_decay']
min_samples = 1e10


def get_kendall(hp: str):
    fixed_params = hparams
    fixed_params.remove(hp)

    run_dict = defaultdict(list)
    gen_dict = defaultdict(list)
    for r in runs:
        if r.state == "running":
            continue

        key = tuple(r.config[x] for x in fixed_params)
        gen = r.summary["generalization/error"]
        gen_dict[key].append(gen)

        run = r.history()
        del run['average_cross_entropy_over_epoch/test']
        run = run.dropna()
        run = run[[c for c in run.columns if c.lower()[:10] == "complexity"]]
        run_dict[key].append(run)

    correlation_dict = defaultdict(lambda: defaultdict(list))
    for key, group in run_dict.items():
        if len(group) < 3:
            continue

        min_samples = min([len(r.index) for r in group])
        sample_measures = {}
        for i in range(min_samples - 1):
            runs_epoch_measures = []
            for run in group:
                runs_epoch_measures.append(run.iloc[i])

            # with each epoch associate a list of the values of the measures at each point
            epoch = group[0].index[i]
            sample_measures[epoch] = pd.DataFrame.from_records(runs_epoch_measures)

        runs_final_measures = []
        for f in group:
            runs_final_measures.append(f.iloc[-1])

        sample_measures["final"] = pd.DataFrame.from_records(runs_final_measures)
        gens = gen_dict[key]

        # Kendall's rank-correlation coefficients
        for k in sample_measures.keys():
            for i in range(len(sample_measures[k].columns)):
                tau_gen_gap, p_value = stats.kendalltau(gens, sample_measures[k].iloc[:, i])
                correlation_dict[sample_measures[k].columns[i]][k].append(tau_gen_gap)

    for k, epochs in correlation_dict.items():
        print("Measure: {}".format(k))
        for e, corrs in epochs.items():
            if len(corrs) < 16:
                continue
            print("Epoch: {}".format(e))
            print("Average correlation: {}".format(sum(corrs)/len(corrs)))


get_kendall("model_depth")
