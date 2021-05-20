#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=lr_test

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=charles.london@wolfson.ox.ac.uk

# run the application
seeds='0 17 43'
lrs='0.1 0.05 0.02'

for s in $seeds; do
  for lr in $lrs; do
    python3 train.py --log_epoch_freq=5 --seed="$s" --lr="$lr" --model_depth="$1" --batch_size="$2" --weight_decay="$3" --batch_norm="$4" --dropout_prob="$5"
  done
done