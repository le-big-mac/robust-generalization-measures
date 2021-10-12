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

module load python3/anaconda
source activate generalization

seeds='0 17 43'
lrs='0.001 0.00158 0.00316 0.00631 0.01 0.02 0.05 0.1'

for s in $seeds; do
  for l in $lrs; do
    python3 train.py --log_epoch_freq=5 --seed="$s" --lr="$l" --model_depth="$3" --batch_size="$4" --weight_decay="$5" --batch_norm="$6" --dropout_prob="$7"
  done
done