#!/bin/bash

data_folder=$HOME/data/crossbeam/arithmetic_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/arithmetic_synthesis/b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python main_arithmetic.py \
    --data_folder $data_folder \
    --eval_every 10000 \
    --gpu 0 \
    --save_dir $save_dir \
    --train_steps 1000000 \
