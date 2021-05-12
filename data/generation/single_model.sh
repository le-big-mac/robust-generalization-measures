#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=12:00:00

# set name of job
#SBATCH --job-name=lr_test

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=charles.london@wolfson.ox.ac.uk

# run the application
lrs='0.001 0.00158 0.00316 0.00631 0.01'

for lr in $lrs; do
  python3 train.py --log_epoch_freq=5 --lr="$lr" --model_depth="$2" --batch_size="$3" --weight_decay="$4" --batch_norm="$5" --dropout_prob="$6"
done