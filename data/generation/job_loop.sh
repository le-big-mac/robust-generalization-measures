#!/bin/bash

# run the application
lrs='0.01'
depth='2'
batch_size='32'
weight_decay='0'

for lr in $lrs; do
  for d in $depth; do
    for b in $batch_size; do
      for w in $weight_decay; do
        sbatch -p small single_model.sh "$lr" "$d" "$b" "$w"
      done
    done
  done
done