#!/bin/bash


data_folder=$HOME/data/crossbeam/tuple_synthesis

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_tuple.py \
    --data_folder $data_folder \
    --gpu_list=0,1,2,3 \
    --num_proc 4 \
    --eval_every 10000 \
    --train_steps 1000000 \
