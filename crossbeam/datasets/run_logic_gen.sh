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

echo "Making manual (hardcoded) tasks"

data_dir=$HOME/data/crossbeam/logic_synthesis_manual

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

seed=10
eval_file=$data_dir/test-tasks.pkl
max_size=20

python3 -m crossbeam.datasets.data_gen \
    --domain=logic  --num_searches 1 \
    --data_gen_seed $seed \
    --output_file $eval_file

echo "Making procedurally generated tasks via bottom-up enumeration"

data_dir=$HOME/data/crossbeam/logic_synthesis_10hr

if [ ! -e $data_dir ];
then
    mkdir -p $data_dir
fi

let time_out=10*60*60 # 10 hours of enumeration

NUM_TRAIN=1000000
NUM_VALID=1000
NUM_TEST=1000

python3 -m crossbeam.datasets.bottom_up_data_generation_logic \
  --max_num_examples=1 --min_num_examples=1 \
  --max_num_inputs=4 --min_num_inputs=4 \
  --max_task_weight $max_size \
  --data_gen_timeout=$time_out \
  --num_tasks_per_split=${NUM_TRAIN} --split_filenames=$data_dir/train-tasks.pkl \
  --num_tasks_per_split=${NUM_VALID} --split_filenames=$data_dir/valid-tasks.pkl \
  --num_tasks_per_split=${NUM_TEST} --split_filenames=$data_dir/test-tasks.pkl
