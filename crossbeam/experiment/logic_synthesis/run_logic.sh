#!/bin/bash

#if you want to do train/test on a handful of tasks that I manually designed
#data_folder=$HOME/data/crossbeam/logic_synthesis_manual
data_folder=$HOME/data/crossbeam/logic_synthesis_10hr

hiddenunits=512
usegreattransformer=0
grad_acc=4
beam_size=4
save_dir=$HOME/results/crossbeam/logic_synthesis/b-$beam_size-h-$hiddenunits-g-$usegreattransformer

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

if [ $usegreattransformer != "0" ];
then
    greatnesscommand="--great_transformer"
else
    greatnesscommand="--nogreat_transformer"
fi

export CUDA_VISIBLE_DEVICES=0

python3 -m crossbeam.experiment.run_crossbeam \
    --domain=logic \
    --model_type=logic \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --beam_size $beam_size \
    --grad_accumulate $grad_acc \
    --max_search_weight 20 \
    --gpu 0 \
    --embed_dim $hiddenunits \
    --eval_every 10000 \
    --num_valid=120 \
    --train_steps 1000000 \
    --train_data_glob train*.pkl \
    --max_num_examples=1 --min_num_examples=1 \
    --max_num_inputs=4 --min_num_inputs=4 \
    $greatnesscommand \
    $@

