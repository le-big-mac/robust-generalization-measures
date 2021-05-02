#!/bin/bash

# run the application
lrs='0.001 0.00158 0.00316 0.00631 0.01'
depth='2 3 4 5 6'
batch_size='32 64 128 256'
weight_decay='0 0.001 0.01 0.1'
batch_norm='True'

for lr in $lrs; do
  for d in $depth; do
    for b in $batch_size; do
      for w in $weight_decay; do
        for bn in $batch_norm; do
          sbatch -p small single_model.sh "$lr" "$d" "$b" "$w" "$bn"
        done
      done
    done
  done
done