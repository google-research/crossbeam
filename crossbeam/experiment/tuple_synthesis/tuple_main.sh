#!/bin/bash

ne=3
ni=3
maxw=6
maxsw=8

data_folder=$HOME/data/crossbeam/tuple_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

beam_size=4
grad_acc=1
save_dir=$HOME/results/crossbeam/tuple_synthesis/b-${beam_size}-g-${grad_acc}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python3 -m crossbeam.experiment.run_crossbeam \
    --domain=tuple \
    --model_type=char \
    --min_num_examples=$ne \
    --max_num_examples=$ne \
    --min_num_inputs=$ni \
    --max_num_inputs=$ni \
    --max_task_weight=$maxw \
    --max_search_weight=$maxsw \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --grad_accumulate $grad_acc \
    --beam_size $beam_size \
    --gpu 0 \
    --eval_every 10000 \
    --train_steps 1000000 \
    $@
