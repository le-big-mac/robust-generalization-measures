def overall_correlation(pairs, weighted=True):
    early_batch_corrs = {}
    epoch_corrs = {}

    for hp_pair, seed_pair_dict in pairs.items():
        run_pair = pairs[hp_pair]
        weight = run_pair.best_weight if weighted else 1

        for seed_pair, run_pair in seed_pair_dict.items():
            if hp_pair[0] == hp_pair[1] and seed_pair[0] >= seed_pair[1]:
                continue

            for step, measure_sign_error in run_pair.early_batches_best_error.items():
                step_corrs, num_pairs = early_batch_corrs.get(step, ({}, 0))

                for measure, error in measure_sign_error.items():
                    corr = step_corrs.get(measure, 0)
                    step_corrs[measure] = corr + weight * error

                early_batch_corrs[step] = (step_corrs, num_pairs + 1)

            for epoch, measure_sign_error in run_pair.epochs_best_error.items():
                e_corrs, num_pairs = epoch_corrs.get(epoch, {})

                for measure, error in measure_sign_error.items():
                    corr = e_corrs.get(measure, 0)
                    e_corrs[measure] = corr + weight * error

                epoch_corrs[epoch] = (e_corrs, num_pairs + 1)

