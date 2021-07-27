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

seeds="0 17 43"
d=$(($3 - 1))
bn_all="$(seq -s ' ' 0 "$d")"

for s in $seeds; do
  python3 train.py --log_epoch_freq=5 --seed="$s" --lr="$2" --model_depth="$3" --batch_size="$4" --weight_decay=0 --bn_layers= --dropout_prob=0
  python3 train.py --log_epoch_freq=5 --seed="$s" --lr="$2" --model_depth="$3" --batch_size="$4" --weight_decay=0 --bn_layers="$bn_all"  --dropout_prob=0
done