#!/bin/bash

data_dir=$HOME/data/crossbeam/tuple_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python tuple_data_gen.py \
    --seed $seed \
    --num_eval 1000 \
    --output_file $eval_file