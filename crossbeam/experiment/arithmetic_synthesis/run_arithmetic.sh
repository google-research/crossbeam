#!/bin/bash

data_folder=$HOME/data/crossbeam/arithmetic_synthesis

export CUDA_VISIBLE_DEVICES=0

#python3 -m crossbeam.experiment.arithmetic_synthesis.main_arithmetic \
python main_arithmetic \
    --data_folder $data_folder \
    --eval_every 10000 \
    --train_steps 1000000 \
    --gpu 0 \
