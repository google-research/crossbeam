#!/bin/bash

data_dir=$HOME/data/crossbeam/arithmetic_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=10 \
    --min_num_examples=3 \
    --max_num_examples=3 \
    --min_num_inputs=3 \
    --max_num_inputs=3 \
    --verbose=False


seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=10 \
    --min_num_examples=3 \
    --max_num_examples=3 \
    --min_num_inputs=3 \
    --max_num_inputs=3 \
    --verbose=False
