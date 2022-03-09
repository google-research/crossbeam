#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

data_folder=$HOME/data/crossbeam/logic_synthesis_dedup

hiddenunits=512
usegreattransformer=1
beam_size=10
grad_acc=4
encode_weight=True
save_dir=$HOME/results/crossbeam/logic_synthesis/b-$beam_size-h-$hiddenunits-g-$usegreattransformer-ew-${encode_weight}

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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m crossbeam.experiment.run_crossbeam \
    --domain=logic \
    --model_type=logic \
    --data_folder $data_folder \
    --save_dir $save_dir \
    --beam_size $beam_size \
    --max_search_weight 20 \
    --gpu_list=0,1,2,3,4,5,6,7 \
    --grad_accumulate $grad_acc \
    --num_proc=8 \
    --encode_weight=$encode_weight \
    --embed_dim $hiddenunits \
    --eval_every 10000 \
    --num_valid=50 \
    --train_steps 1000000 \
    --train_data_glob train*.pkl \
    --max_num_examples=1 --min_num_examples=1 \
    --max_num_inputs=4 --min_num_inputs=4 \
    $greatnesscommand \
    $@

