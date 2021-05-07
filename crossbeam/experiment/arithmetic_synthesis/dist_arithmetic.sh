#!/bin/bash

data_folder=$HOME/data/crossbeam/arithmetic_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/arithmetic_synthesis/dist-b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_arithmetic.py \
    --data_folder $data_folder \
    --eval_every 10000 \
    --gpu_list 0,1,2,3 \
    --save_dir $save_dir \
    --num_proc 4 \
    --train_steps 1000000 \
    --port 29501 \
    --model_type int_op \
