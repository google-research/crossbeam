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

from absl import flags
import torch
import random
import numpy as np

flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_string('train_data_glob', None, 'offline data dumps')
flags.DEFINE_string('test_data_glob', None, 'test data dumps')
flags.DEFINE_boolean('use_ur', True, 'use UR for evaluation?')
flags.DEFINE_boolean('score_normed', True, 'whether to normalize the score into valid probability')
flags.DEFINE_boolean('type_masking', True, 'use type masking during synthesis?')
flags.DEFINE_integer('grad_accumulate', 1, '# forward / backward steps')
flags.DEFINE_integer('max_search_weight', 12, '')
flags.DEFINE_integer('num_valid', -1, 'num tasks for evaluation per process during training')
flags.DEFINE_float('timeout', 5, 'time limit in seconds')
flags.DEFINE_integer('max_values_explored', None, 'max number of values to explore per search')

flags.DEFINE_string('io_encoder', 'char', 'io encoder, choose from [char, signature, char_signature]')
flags.DEFINE_string('value_encoder', 'char', 'value encoder, choose from [char, signature, char_signature]')
flags.DEFINE_boolean('encode_weight', False, 'encode value weights?')
flags.DEFINE_boolean('static_weight', False, 'static weights?')


def set_global_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
