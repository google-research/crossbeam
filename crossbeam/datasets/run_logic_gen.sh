#!/bin/bash

data_dir=$HOME/data/crossbeam/logic_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=logic \
    --seed $seed \
    --output_file $eval_file #--num_eval 1000 \


seed=10
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=logic \
    --seed $seed \
    --output_file $eval_file #--num_eval 1000 \
