#!/bin/bash


data_folder=$HOME/data/crossbeam/arithmetic_synthesis

export CUDA_VISIBLE_DEVICES=1

python main_arithmetic.py \
    --data_folder $data_folder \
    --eval_every 10000 \
    --train_steps 1000000 \
    --gpu 0 \
