#!/bin/bash


data_folder=$HOME/data/crossbeam/arithmetic_synthesis

export CUDA_VISIBLE_DEVICES=0

python main_arithmetic.py \
    --data_folder $data_folder \
    --gpu -1 \
