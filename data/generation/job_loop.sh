#!/bin/bash

# run the application
lrs='0.01'
depth='2'
batch_size='32'
weight_decay='0'
batch_norm='True'
dropout_prob='0'

for lr in $lrs; do
  for d in $depth; do
    for b in $batch_size; do
      for w in $weight_decay; do
        for bn in $batch_norm; do
          for p in $dropout_prob; do
            sbatch -p small single_model.sh "$lr" "$d" "$b" "$w" "$bn" "$p"
          done
        done
      done
    done
  done
done