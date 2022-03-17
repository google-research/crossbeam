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

import os
import pickle5 as pkl
import sys

from crossbeam.datasets import logic_data

data_file = sys.argv[1]
output_file = sys.argv[2]
with open(data_file, 'rb') as f:
  data = pkl.load(f)

print('Original data length: {}'.format(len(data)))

logic_constants, logic_operations = logic_data.get_consts_and_ops()
manual_tasks = logic_data.all_manual_logic_tasks(logic_operations)


solutions_to_indices = {task.solution: i for i, task in enumerate(data)}
bad_indices = []

for task in manual_tasks:
  manual_solution = task.solution
  if manual_solution in solutions_to_indices:
    index = solutions_to_indices[manual_solution]
    print('Found task in training data, index {}: {}'.format(index, task))
    bad_indices.append(index)

for i in sorted(bad_indices, reverse=True):
  print('Removing task index {}'.format(i))
  del data[i]

print('Filtered data length: {}'.format(len(data)))

with open(output_file, 'wb') as f:
  pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
