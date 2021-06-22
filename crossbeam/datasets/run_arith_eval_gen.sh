#!/bin/bash

ne=3
ni=3
maxw=8

data_dir=$HOME/data/crossbeam/arithmetic_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --seed=$seed \
    --num_tasks=1000 \
    --num_examples=$ne \
    --num_inputs=$ni \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --verbose=False


seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --seed=$seed \
    --num_tasks=1000 \
    --num_examples=$ne \
    --num_inputs=$ni \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --verbose=False
