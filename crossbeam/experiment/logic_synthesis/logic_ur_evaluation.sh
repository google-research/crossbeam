#!/bin/bash

tout=120
maxw=10
maxne=4
maxni=3
maxsw=20

data_folder=$HOME/data/crossbeam/logic_synthesis_manual

beam_size=10
grad_acc=1
save_dir=$HOME/results/crossbeam/logic_synthesis

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=3


python3 -m crossbeam.experiment.run_crossbeam \
    --domain=logic \
    --model_type=logic \
    --min_num_examples=1 \
    --max_num_examples=1 \
    --min_num_inputs=4 \
    --max_num_inputs=4 \
    --max_search_weight=$maxsw \
    --data_folder=$data_folder \
    --save_dir=$save_dir \
    --load_model=mlp/model-best-valid.ckpt \
    --grad_accumulate=$grad_acc \
    --beam_size=$beam_size \
    --gpu_list=0 \
    --num_proc=1 \
    --eval_every=1 \
    --encode_weight=True \
    --train_steps=0 \
    --random_beam=False \
    --do_test=True \
    --use_ur=True \
    --nogreat_transformer \
    --timeout=30 \
    $@
