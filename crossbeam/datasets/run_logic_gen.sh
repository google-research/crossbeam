#!/bin/bash

echo "Making manual (hardcoded) tasks"

data_dir=$HOME/data/crossbeam/logic_synthesis_manual

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl



python data_gen.py \
    --domain=logic \
    --data_gen_seed $seed \
    --output_file $eval_file #--num_eval 1000 \


seed=10
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=logic \
    --data_gen_seed $seed \
    --output_file $eval_file #--num_eval 1000

echo "Making procedurally generated tasks via bottom-up enumeration"


data_dir=$HOME/data/crossbeam/logic_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

echo "making testing"
file=$data_dir/test-tasks.pkl
python bottom_up_data_generation.py --domain=logic --max_num_examples=1 --min_num_examples=1 --max_num_inputs=4 --min_num_inputs=4 --output_file $file

echo "making validation"
file=$data_dir/valid-tasks.pkl
python bottom_up_data_generation.py --domain=logic --max_num_examples=1 --min_num_examples=1 --max_num_inputs=4 --min_num_inputs=4 --output_file $file

echo "making training"
file=$data_dir/train-tasks.pkl
python bottom_up_data_generation.py --domain=logic --max_num_examples=1 --min_num_examples=1 --max_num_inputs=4 --min_num_inputs=4 --output_file $file
