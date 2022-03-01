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
maxsw=20

data_folder=$HOME/crossbeam/crossbeam/data/sygus
#data_folder=$HOME/crossbeam/crossbeam/data/new

beam_size=10
grad_acc=1
save_dir=$HOME/results/crossbeam/bustle/

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
    --grad_accumulate=$grad_acc \
    --beam_size=$beam_size \
    --gpu_list=0 \
    --num_proc=1 \
    --eval_every=1 \
    --train_steps=0 \
    --train_data_glob=train-tasks*.pkl \
    --random_beam=False \
    --do_test=True \
    --use_ur=True \
    --timeout=600 \
    --max_values_explored=50000 \
    $@
    #--load_model=io-bustle_sig-value-bustle_sig-b-4-g-4/model-best-valid.ckpt \
