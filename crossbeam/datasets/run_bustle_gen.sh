#!/bin/bash

data_dir=$HOME/data/crossbeam/bustle

seed=0
tout=120
maxw=10
maxne=4
maxni=3
num_proc=90
out_dir=$data_dir/t-${tout}-maxw-${maxw}-maxne-${maxne}-maxni-${maxni}

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

training_file=$out_dir/train-tasks.pkl

python3 -m crossbeam.datasets.bottom_up_data_generation \
    --domain=bustle \
    --output_file=$training_file \
    --data_gen_seed=$seed \
    --data_gen_timeout=$tout \
    --num_tasks=1000 \
    --num_searches=10000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=2 \
    --max_num_examples=$maxne \
    --min_num_inputs=1 \
    --max_num_inputs=$maxni \
    --num_datagen_proc=$num_proc \
    --verbose=False
