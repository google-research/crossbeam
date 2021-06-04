#!/bin/bash

data_folder=$HOME/data/crossbeam/arithmetic_synthesis

beam_size=4
save_dir=$HOME/results/crossbeam/arithmetic_synthesis/dist-b-$beam_size

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

#python3 -m crossbeam.experiment.run_crossbeam \
python run_crossbeam.py \
    --domain=arithmetic \
    --model_type=int \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --gpu_list 0,1,2,3 \
    --num_proc 4 \
    --eval_every 10000 \
    --train_steps 1000000 \
    --port 29501 \
    $@
