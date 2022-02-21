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
maxw=10

data_dir=$HOME/data/crossbeam/arithmetic_synthesis/ne-${ne}-ni-${ni}-maxw-${maxw}

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=1
eval_file=$data_dir/valid-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=$ne \
    --max_num_examples=$ne \
    --min_num_inputs=$ni \
    --max_num_inputs=$ni \
    --verbose=False


seed=10
eval_file=$data_dir/test-tasks.pkl

python data_gen.py \
    --domain=arithmetic \
    --output_file=$eval_file \
    --data_gen_seed=$seed \
    --num_tasks=1000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=$ne \
    --max_num_examples=$ne \
    --min_num_inputs=$ni \
    --max_num_inputs=$ni \
    --verbose=False
