#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ne=4
ni=3
tout=10
maxw=10

data_folder=$HOME/data/crossbeam/bustle/t-${tout}-maxw-${maxw}-maxne-${maxne}-maxni-${maxni}

embed=128
bsize=512
rnn_layers=3
beam_size=4
save_dir=$HOME/results/crossbeam/robustfill/bustle/e-${embed}-b-${bsize}-r-${rnn_layers}-maxw-${maxw}-t-${tout}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m crossbeam.robustfill.main_robustfill \
    --gpu 0 \
    --domain=bustle \
    --min_num_examples=2 \
    --max_num_examples=$ne \
    --min_num_inputs=2 \
    --max_num_inputs=$ni \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --decoder_rnn_layers $rnn_layers \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --lr 1e-3 \
    --embed_dim $embed \
    --beam_size $beam_size \
    --eval_every 2000 \
    --train_steps 1000000 \
    --batch_size $bsize \
    --n_para_dataload 4 \
    --train_data_glob train-tasks*.pkl \
    $@
