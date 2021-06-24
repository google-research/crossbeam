#!/bin/bash

data_dir=$HOME/data/crossbeam/bustle

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
training_file=$data_dir/train-tasks-${seed}.pkl

python3 crossbeam/datasets/bottom_up_data_generation.py \
    --domain=bustle \
    --output_file=$training_file \
    --data_gen_seed=$seed \
    --data_gen_timeout=120 \
    --num_tasks=1000 \
    --num_searches=500 \
    --min_task_weight=3 \
    --max_task_weight=10 \
    --min_num_examples=2 \
    --max_num_examples=4 \
    --min_num_inputs=1 \
    --max_num_inputs=3 \
    --verbose=False
