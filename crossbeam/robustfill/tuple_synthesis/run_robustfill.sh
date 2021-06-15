#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ne=3
ni=3
maxw=6

data_folder=$HOME/data/crossbeam/tuple_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

embed=128
bsize=512
rnn_layers=3
beam_size=4
save_dir=$HOME/results/crossbeam/robustfill/tuple_synthesis/e-$embed-b-$bsize-r-$rnn_layers

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_robustfill.py \
    --gpu 0 \
    --num_examples=$ne \
    --num_inputs=$ni \
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
    --n_para_dataload 8 \
