#!/bin/bash

ne=3
ni=3
maxw=8

data_dir=$HOME/data/crossbeam/tuple_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=tuple \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=$ne \
    --max_num_examples=$ne \
    --min_num_inputs=$ni \
    --max_num_inputs=$ni \
    --verbose=False


seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=tuple \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=$ne \
    --max_num_examples=$ne \
    --min_num_inputs=$ni \
    --max_num_inputs=$ni \
    --verbose=False
