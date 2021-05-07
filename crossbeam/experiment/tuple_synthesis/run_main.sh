#!/bin/bash

data_folder=$HOME/data/crossbeam/tuple_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/tuple_synthesis/b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

#python3 -m crossbeam.experiment.tuple_synthesis.main_tuple \
python main_tuple \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --beam_size $beam_size \
    --gpu 0 \
    --eval_every 10000 \
    --train_steps 1000000 \
