#!/bin/bash

tout=10
maxw=10
maxne=4
maxni=3
maxsw=12

data_folder=$HOME/data/crossbeam/bustle/t-${tout}-maxw-${maxw}-maxne-${maxne}-maxni-${maxni}

beam_size=4
grad_acc=4
save_dir=$HOME/results/crossbeam/bustle/b-${beam_size}-g-${grad_acc}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
    --data_folder $data_folder \
    --save_dir $save_dir \
    --grad_accumulate $grad_acc \
    --beam_size $beam_size \
    --gpu_list=0,1,2,3,4,5,6,7 \
    --num_proc=8 \
    --embed_dim=512 \
    --eval_every 10000 \
    --use_ur=False \
    --train_steps 1000000 \
    --train_data_glob train-tasks*.pkl \
    --random_beam=False \
    $@
