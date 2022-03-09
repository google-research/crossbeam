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

ne=3
ni=3
maxw=6
maxsw=8

data_folder=$HOME/data/crossbeam/tuple_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

beam_size=4
grad_acc=3
save_dir=$HOME/results/crossbeam/tuple_synthesis/dist-b-${beam_size}-g-${grad_acc}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
    --gpu_list=0,1,2,3,4,5,6,7 \
    --num_proc=8 \
    --eval_every 10000 \
    --train_steps 1000000 \
    $@
