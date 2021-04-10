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

from crossbeam.unique_randomizer import unique_randomizer as ur

# TODO: make actual tests out of this haha
randomizer = ur.UniqueRandomizer()

distributions = {
    '': [0.2, 0.1, 0.6],
    '0': [0.1, 0.1, 0.4],
    '1': [0.3, 0.7, 0.2],
    '2': [0.2, 0.5, 0.1],
}

for i in range(8):
  seq = ''
  for _ in range(2):
    choice = randomizer.sample_distribution([0.5, 0.2, 0.1])
    seq += str(choice)
  randomizer.mark_sequence_complete()

  print('Sample {}: {}'.format(i, seq))

print('\nAdding more options!\n')

distributions = {
    '': [0.2, 0.1, 0.6, 0.2, 0.1],
    '0': [0.1, 0.1, 0.4, 0.7, 0.3],
    '1': [0.3, 0.7, 0.2, 0.1, 0.1],
    '2': [0.2, 0.5, 0.1, 0.3, 0.4],
    '3': [0.1, 0.4, 0.2, 0.3, 0.1],
    '4': [0.2, 0.1, 0.2, 0.7, 0.1],
}


for i in range(17):
  seq = ''
  for _ in range(2):
    choice = randomizer.sample_distribution([0.5, 0.2, 0.1, 0.7, 0.2])
    seq += str(choice)
  randomizer.mark_sequence_complete()

  print('Sample {}: {}'.format(i, seq))
