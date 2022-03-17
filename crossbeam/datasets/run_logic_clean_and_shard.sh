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

data_root=$HOME/data/crossbeam/logic_synthesis_10hr
output_root=$HOME/data/crossbeam/logic_synthesis_dedup

if [ ! -e $output_root ];
then
	mkdir -p $output_root
fi

python clean_logic.py \
  $data_root/train-tasks.pkl \
  $output_root/train-tasks.pkl \

python make_shards.py \
  $output_root

cp $data_root/valid-tasks.pkl $output_root
