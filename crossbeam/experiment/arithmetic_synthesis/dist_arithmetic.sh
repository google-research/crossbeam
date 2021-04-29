#!/bin/bash


data_folder=$HOME/data/crossbeam/arithmetic_synthesis

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_arithmetic.py \
    --data_folder $data_folder \
    --eval_every 10000 \
    --gpu_list 0,1,2,3 \
    --num_proc 4 \
    --train_steps 1000000 \
