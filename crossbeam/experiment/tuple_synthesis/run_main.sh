#!/bin/bash


data_folder=$HOME/data/crossbeam/tuple_synthesis

export CUDA_VISIBLE_DEVICES=0

python main_tuple.py \
    --data_folder $data_folder \
    --eval_every 10000 \
    --train_steps 1000000 \
    --gpu 0 \
