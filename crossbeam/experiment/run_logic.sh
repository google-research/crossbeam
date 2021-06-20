#!/bin/bash

data_folder=$HOME/data/crossbeam/logic_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/logic_synthesis/b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python run_crossbeam.py \
    --domain=logic \
    --model_type=logic \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --beam_size $beam_size \
    --max_search_weight 20 \
    --gpu 0 \
    --eval_every 10000 \
    --train_steps 1000000 \
    --memorize \
    $@

