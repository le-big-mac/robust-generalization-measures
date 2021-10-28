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

python3 train.py --log_epoch_freq=5 --epochs=1000 --seed="$1" --lr="$2" --model_depth="$3" --batch_size="$4" --weight_decay="$5" --batch_norm="$6" --dropout_prob="$7"