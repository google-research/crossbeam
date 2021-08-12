#!/bin/bash

tout=120
maxw=10
maxne=4
maxni=3
maxsw=20

data_folder=$HOME/crossbeam/crossbeam/data

beam_size=10
grad_acc=1
save_dir=$HOME/results/crossbeam/bustle/ur-test

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0


python3 -m crossbeam.experiment.run_crossbeam \
    --domain=bustle \
    --model_type=char \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=2 \
    --max_num_examples=$maxne \
    --min_num_inputs=1 \
    --max_num_inputs=$maxni \
    --max_search_weight=$maxsw \
    --data_folder=$data_folder \
    --save_dir=$save_dir \
    --load_model=saved/model-best-valid.ckpt \
    --grad_accumulate=$grad_acc \
    --beam_size=$beam_size \
    --gpu_list=0 \
    --num_proc=1 \
    --eval_every=1 \
    --train_steps=0 \
    --train_data_glob=train-tasks*.pkl \
    --random_beam=False \
    --do_test=True \
    --timeout=30 \
    $@
