#!/bin/bash

echo "Making manual (hardcoded) tasks"

data_dir=$HOME/data/crossbeam/logic_synthesis_manual

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl
max_size=15
let time_out=60*60 # an hour of enumeration

python data_gen.py \
    --domain=logic  --num_searches 1\
    --data_gen_seed $seed \
    --output_file $eval_file #--num_eval 1000 \


seed=10
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=logic --num_searches 1\
    --data_gen_seed $seed \
    --output_file $eval_file #--num_eval 1000

echo "Making procedurally generated tasks via bottom-up enumeration"


data_dir=$HOME/data/crossbeam/logic_synthesis

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

NT=1000000 # number of training tasks
for fn in $data_dir/train-tasks.pkl $data_dir/test-tasks.pkl $data_dir/valid-tasks.pkl; do
    echo "producing output in" $fn
    
    python bottom_up_data_generation.py --domain=logic  --num_searches 1 --max_num_examples=1 --min_num_examples=1 --max_num_inputs=4 --min_num_inputs=4 --output_file $fn --max_task_weight $max_size --data_gen_timeout=$time_out  --num_tasks=$NT

    NT=1000 # number of testing and validation tasks
done
