#!/bin/bash

# run the application
seeds='0'
lrs='0.01'
depth='2 3 4 5'
batch_size='32'
weight_decay='0'
batch_norm='True'
dropout_prob='0.4'

for s in $seeds; do
  for lr in $lrs; do
    for d in $depth; do
      for b in $batch_size; do
        for w in $weight_decay; do
          for bn in $batch_norm; do
            for p in $dropout_prob; do
              sbatch -p small single_model.sh "$s" "$lr" "$d" "$b" "$w" "$bn" "$p"
            done
          done
        done
      done
    done
  done
done