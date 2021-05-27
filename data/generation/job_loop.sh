#!/bin/bash

# run the application
seeds='0 17 43'
lrs='0.1 0.05 0.02'
depth='2'
batch_size='32'
weight_decay='0'
batch_norm='False'
dropout_prob='0 0.2 0.5 0.7'

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