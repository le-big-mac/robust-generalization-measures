import wandb
import pandas as pd
import scipy.stats as stats

api = wandb.Api()
runs = api.runs("rgm")

gens = []
frames = []
min_samples = 1e10

for r in runs:
    run = r.history()
    run = run.dropna()
    run = run[[c for c in run.columns if c.lower()[:10] == "complexity"]]
    min_samples = min(len(run.index), min_samples)
    frames.append(run)
    gens.append(r.summary["generalization/error"])

print(gens)

sample_measures = {}
for i in range(min_samples-1):
    rows = []
    for f in frames:
        rows.append(f.iloc[i])
    sample_measures[frames[0].index[i]] = pd.DataFrame.from_records(rows)

rows = []
for f in frames:
    rows.append(f.iloc[-1])

sample_measures["final"] = pd.DataFrame.from_records(rows)
print(sample_measures)

# Kendall's rank-correlation coefficients
print('Correlations Between Compexity Measures and Generalization Gap (Test Error-Train Error):\n')

for k in sample_measures.keys():
    print("Epoch: {}".format(k))
    for i in range(len(sample_measures[k].columns)):
        tau_gen_gap, p_value = stats.kendalltau(gens, sample_measures[k].iloc[:, i])
        print(sample_measures[k].columns[i], '\t\t', tau_gen_gap)
