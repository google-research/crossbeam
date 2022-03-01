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

tout=120
maxw=10
maxne=4
maxni=3
maxsw=12

data_folder=$HOME/data/crossbeam/bustle/t-${tout}-maxw-${maxw}-maxne-${maxne}-maxni-${maxni}

beam_size=10
grad_acc=4
io=bustle_sig
value=bustle_sig

save_dir=$HOME/results/crossbeam/bustle/vw-tout-${tout}-io-${io}-value-${value}-b-${beam_size}-g-${grad_acc}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m crossbeam.experiment.run_crossbeam \
    --domain=bustle \
    --model_type=char \
    --io_encoder=${io} \
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
    --gpu_list=0,1,2,3,4,5,6,7 \
    --num_proc=8 \
    --embed_dim=512 \
    --eval_every 10000 \
    --use_ur=False \
    --encode_weight=True \
    --train_steps 1000000 \
    --train_data_glob train-tasks*.pkl \
    --random_beam=False \
    $@
