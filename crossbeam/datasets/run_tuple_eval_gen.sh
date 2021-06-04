#!/bin/bash

data_dir=$HOME/data/crossbeam/tuple_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=tuple \
    --output_file=$eval_file \
    --seed=$seed \
    --num_eval=1000 \
    --num_examples=3 \
    --num_inputs=3 \
    --min_task_weight=3 \
    --max_task_weight=10 \
    --verbose=False


seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=tuple \
    --output_file=$eval_file \
    --seed=$seed \
    --num_eval=1000 \
    --num_examples=3 \
    --num_inputs=3 \
    --min_task_weight=3 \
    --max_task_weight=10 \
    --verbose=False
