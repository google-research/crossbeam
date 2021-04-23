#!/bin/bash

data_dir=$HOME/data/crossbeam/tuple_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl

python tuple_data_gen.py \
    --seed $seed \
    --output_file $eval_file
