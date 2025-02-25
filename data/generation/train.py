#!/usr/bin/env python3
import pickle

import simple_parsing
import torch

from experiment import Experiment
from experiment_config import Config, HParams, State
from logs import WandbLogger

if __name__ == '__main__':
    # Prepare experiment settings
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(HParams, dest="hparams")
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(State, dest="state")

    args = parser.parse_args()
    hparams: HParams = args.hparams
    print(vars(hparams))
    config: Config = args.config
    config = Config(config.id, config.log_batch_freq, config.log_epoch_freq, config.log_steps, config.save_state_epochs,
                    config.root_dir / hparams.wandb_md5, config.data_dir, config.verbosity, config.use_tqdm)
    config.setup_dirs()
    state: State = args.state
    state.ce_check_milestones = hparams.ce_target_milestones.copy()

    # Run experiment
    device = torch.device('cuda' if hparams.use_cuda else 'cpu')
    logger = WandbLogger('default', hparams.to_tensorboard_dict(), hparams.wandb_md5)


    def dump_results(epoch, val_eval, train_eval):
        results = {
            'state': state,
            'hparams': hparams,
            'final_results_val': val_eval,
            'final_results_train': train_eval,
        }
        with open(config.results_dir / '{}.pkl'.format(config.id, epoch), mode='wb') as results_file:
            pickle.dump(results, results_file)


    Experiment(state, device, hparams, config, logger, dump_results).train()
