#!/bin/bash


data_folder=$HOME/data/crossbeam/tuple_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/tuple_synthesis/dist-b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_tuple.py \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --beam_size $beam_size \
    --gpu_list=0,1,2,3 \
    --num_proc 4 \
    --eval_every 10000 \
    --train_steps 1000000 \
