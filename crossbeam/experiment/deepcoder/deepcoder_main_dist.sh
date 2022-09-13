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


tout=60
maxw=14
maxne=5
maxni=3
skip=0.00
lambdaskip=0.00

data_folder=$HOME/xlambda-data/deepcoder/t-${tout}-maxne-${maxne}-maxni-${maxni}-skip-${skip}-lambdaskip-${lambdaskip}

beam_size=10
grad_acc=4
maxsw=12
io=lambda_signature
value=lambda_signature

save_dir=$HOME/results/xlambda/deepcoder/tout-${tout}-io-${io}-value-${value}-b-${beam_size}-g-${grad_acc}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m crossbeam.experiment.run_crossbeam \
    --domain=deepcoder \
    --io_encoder=${io} \
    --model_type=deepcoder \
    --value_encoder=${value} \
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
    --num_proc=8 \
    --gpu_list=0,1,2,3,4,5,6,7 \
    --embed_dim=64 \
    --eval_every 10000 \
    --use_ur=False \
    --encode_weight=True \
    --train_steps 1000000 \
    --train_data_glob train-tasks*.pkl \
    --random_beam=False \
    --lr=1e-4 \
    $@
