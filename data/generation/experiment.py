from copy import deepcopy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from dataset_helpers import get_dataloaders
from experiment_config import (
    Config,
    DatasetSubsetType,
    HParams,
    State,
    EvaluationMetrics,
    OptimizerType,
)
from logs import BaseLogger, Printer
from measures import get_all_measures
from models import NiN
from experiment_config import ComplexityType as CT


class Experiment:
    def __init__(
            self,
            state: State,
            device: torch.device,
            hparams: HParams,
            config: Config,
            logger: BaseLogger,
            result_save_callback: Optional[object] = None
    ):
        self.state = state
        self.device = device
        self.hparams = hparams
        self.config = config

        # Random Seeds
        random.seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        torch.cuda.manual_seed_all(self.hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Logging
        self.logger = logger
        # Printing
        self.printer = Printer(self.config.id, self.config.verbosity)
        self.result_save_callback = result_save_callback

        # Model
        self.model = NiN(self.hparams.model_depth, self.hparams.model_width, self.hparams.base_width,
                         self.hparams.batch_norm, self.hparams.dropout_prob, self.hparams.dataset_type)
        print(self.model)
        print("Number of parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.model.to(device)
        self.init_model = deepcopy(self.model)

        # Optimizer
        if self.hparams.optimizer_type == OptimizerType.SGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr,
                                             weight_decay=hparams.weight_decay)
        elif self.hparams.optimizer_type == OptimizerType.SGD_MOMENTUM:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9,
                                             weight_decay=hparams.weight_decay)
        elif self.hparams.optimizer_type == OptimizerType.ADAM:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr,
                                              weight_decay=hparams.weight_decay)
        else:
            raise KeyError

        # Load data
        self.train_loader, self.train_eval_loader, self.test_loader = get_dataloaders(self.hparams, self.config,
                                                                                      self.device)
        self.train_history = []

    def save_state(self, file_name: str = '') -> None:
        checkpoint_file = self.config.checkpoint_dir / (file_name + '.pt')
        torch.save({
            'config': self.hparams,
            'state': self.state,
            'model': self.model.state_dict(),
            'init_model': self.init_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'np_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }, checkpoint_file)

    def _train_epoch(self) -> None:
        self.model.train()
        ce_check_batches = [(len(self.train_loader) // (2 ** (self.state.ce_check_freq))) * (i + 1) for i in
                            range(2 ** (self.state.ce_check_freq) - 1)]
        ce_check_batches.append(len(self.train_loader) - 1)

        batch_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.state.batch = batch_idx
            self.state.global_batch = (self.state.epoch - 1) * len(self.train_loader) + self.state.batch

            if self.state.global_batch in self.config.log_steps:
                train_eval = self.evaluate_batch(DatasetSubsetType.TRAIN, True, batch_losses)
                val_eval = self.evaluate_batch(DatasetSubsetType.TEST)
                train_eval.all_complexities[CT.VALIDATION_ACC] = val_eval.acc
                self.logger.log_generalization_gap(self.state, train_eval.acc, val_eval.acc, train_eval.avg_loss,
                                                   val_eval.avg_loss, train_eval.all_complexities)
            if self.state.global_batch > max(self.config.log_steps):
                break

            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(data)
            cross_entropy = F.cross_entropy(logits, target)

            cross_entropy.backward()
            loss = cross_entropy.clone()
            batch_losses.append(loss)

            self.model.train()

            self.optimizer.step()

            # Log everything
            self.printer.batch_end(self.config, self.state, data, self.train_loader, loss)
            self.logger.log_batch_end(self.config, self.state, cross_entropy, loss)

            # Cross-entropy stopping check
            if batch_idx == ce_check_batches[0]:
                ce_check_batches.pop(0)
                dataset_ce = self.evaluate_cross_entropy(DatasetSubsetType.TRAIN)[0]
                if dataset_ce < self.hparams.ce_target:
                    self.state.converged = True
                else:
                    while len(self.state.ce_check_milestones) > 0 and dataset_ce <= self.state.ce_check_milestones[0]:
                        passed_milestone = self.state.ce_check_milestones[0]
                        print(f'passed ce milestone {passed_milestone}')
                        self.state.ce_check_milestones.pop(0)
                        self.state.ce_check_freq += 1
                        if self.config.save_state_epochs is not None:
                            self.save_state(f'_ce_{passed_milestone}')

            if self.state.converged:
                break

        self.train_history.append(sum(batch_losses)/len(batch_losses))
        self.evaluate_cross_entropy(DatasetSubsetType.TRAIN, True)
        self.evaluate_cross_entropy(DatasetSubsetType.TEST, True)

    def train(self) -> None:
        self.printer.train_start(self.device)
        train_eval, val_eval = None, None

        self.state.global_batch = 0
        for epoch in trange(self.state.epoch, self.hparams.epochs + 1, disable=(not self.config.use_tqdm)):
            self.state.epoch = epoch
            self._train_epoch()

            if self.state.global_batch > max(self.config.log_steps):
                break

            is_evaluation_epoch = (
                        epoch == 1 or epoch == self.hparams.epochs or epoch % self.config.log_epoch_freq == 0)
            if is_evaluation_epoch or self.state.converged:
                train_eval = self.evaluate(DatasetSubsetType.TRAIN, True)
                val_eval = self.evaluate(DatasetSubsetType.TEST)
                train_eval.all_complexities[CT.VALIDATION_ACC] = val_eval.acc

                self.logger.log_generalization_gap(self.state, train_eval.acc, val_eval.acc, train_eval.avg_loss,
                                                   val_eval.avg_loss, train_eval.all_complexities)
                self.printer.epoch_metrics(epoch, train_eval, val_eval)

                if self.state.epoch > 300 and val_eval.acc > 0.99:
                    self.state.converged = True

            if epoch == self.hparams.epochs or self.state.converged:
                self.result_save_callback(epoch, val_eval, train_eval)

            # Save state
            is_save_epoch = self.config.save_state_epochs is not None and (
                    epoch in self.config.save_state_epochs or epoch == self.hparams.epochs or self.state.converged)
            if is_save_epoch:
                self.save_state(f"epoch_{epoch}")

            if self.state.converged:
                print('Converged')
                break

        self.printer.train_end()

        # if train_eval is None or val_eval is None:
        #     raise RuntimeError

    @torch.no_grad()
    def evaluate_batch(self, dataset_subset_type: DatasetSubsetType, compute_all_measures: bool = False,
                       batches=None) -> EvaluationMetrics:
        self.model.eval()
        data_loader = [self.train_eval_loader, self.test_loader][dataset_subset_type]

        cross_entropy_loss, acc, num_correct = self.evaluate_cross_entropy(dataset_subset_type)

        all_complexities = {}
        if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
            all_complexities = get_all_measures(self.model, self.init_model, data_loader, acc, self.train_history,
                                                self.hparams.seed)
            all_complexities[CT.SOTL] = sum(batches)
            all_complexities[CT.SOTL_10] = sum(batches[-10::])

        self.logger.log_epoch_end(self.hparams, self.state, dataset_subset_type, cross_entropy_loss, acc, 0)

        return EvaluationMetrics(acc, cross_entropy_loss, num_correct, len(data_loader.dataset), all_complexities)

    @torch.no_grad()
    def evaluate(self, dataset_subset_type: DatasetSubsetType, compute_all_measures: bool = False) -> EvaluationMetrics:
        self.model.eval()
        data_loader = [self.train_eval_loader, self.test_loader][dataset_subset_type]

        cross_entropy_loss, acc, num_correct = self.evaluate_cross_entropy(dataset_subset_type)

        all_complexities = {}
        if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
            all_complexities = get_all_measures(self.model, self.init_model, data_loader, acc, self.train_history,
                                                self.hparams.seed)

        self.logger.log_epoch_end(self.hparams, self.state, dataset_subset_type, cross_entropy_loss, acc,
                                  self.train_history[-1] if dataset_subset_type == DatasetSubsetType.TRAIN else None)

        return EvaluationMetrics(acc, cross_entropy_loss, num_correct, len(data_loader.dataset), all_complexities)

    @torch.no_grad()
    def evaluate_cross_entropy(self, dataset_subset_type: DatasetSubsetType, log: bool = False):
        self.model.eval()
        cross_entropy_loss = 0
        num_correct = 0

        data_loader = [self.train_eval_loader, self.test_loader][dataset_subset_type]
        num_to_evaluate_on = len(data_loader.dataset)

        for data, target in data_loader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            logits = self.model(data)
            cross_entropy = F.cross_entropy(logits, target, reduction='sum')
            cross_entropy_loss += cross_entropy.item()  # sum up batch loss

            pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            num_correct += batch_correct.sum()

        cross_entropy_loss /= num_to_evaluate_on
        acc = num_correct.item() / num_to_evaluate_on

        if log:
            self.logger.log_epoch_end(self.hparams, self.state, dataset_subset_type, cross_entropy_loss, acc,
                                      None if (dataset_subset_type != DatasetSubsetType.TRAIN or len(self.train_history) == 0)
                                      else self.train_history[-1])
        print("Cross entropy loss: {}".format(cross_entropy_loss))
        print("Accuracy: {}".format(acc))
        print("Num correct: {}".format(num_correct))
        return cross_entropy_loss, acc, num_correct
