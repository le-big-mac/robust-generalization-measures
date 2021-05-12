#!/bin/bash

# run the application
depth='2 3 4 5'
batch_size='32 64 128 256'
weight_decay='0'
batch_norm='True'
dropout_prob='0'

for d in $depth; do
  for b in $batch_size; do
    for w in $weight_decay; do
      for bn in $batch_norm; do
        for p in $dropout_prob; do
          sbatch -p small single_model.sh "$d" "$b" "$w" "$bn" "$p"
        done
      done
    done
  done
done