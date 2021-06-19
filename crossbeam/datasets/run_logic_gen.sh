#!/bin/bash

data_dir=$HOME/data/crossbeam/logic_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl

python logic_data_generator.py \
    --data_gen_seed $seed \
    --output_file $eval_file #--num_eval 1000 \
