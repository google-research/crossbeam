# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import flags
import torch

"""
We define a set of args here to avoid repetitions in each main function.

The set of args included here should be consistent in different scenarios and should have no ambiguity.
"""

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string('data_folder', None, 'folder for offline data dump')
flags.DEFINE_string('save_dir', None, 'folder for saving model dump/logs')
flags.DEFINE_string('load_model', None, 'load saved model from $save_dir/$load_model')
flags.DEFINE_bool('do_test', False, 'eval against test set?')
flags.DEFINE_string('json_results_file', None, 'Where to write JSON results')

# nn configs
flags.DEFINE_integer('batch_size', 32, 'minibatch size')
flags.DEFINE_integer('embed_dim', 512, 'embedding dimension')
flags.DEFINE_integer('n_para_dataload', 0, 'num of parallel data loader')
flags.DEFINE_integer('decoder_rnn_layers', 3, '# rnn autoregressive model decoder layers')
flags.DEFINE_float('grad_clip', 5.0, 'clip grad')
flags.DEFINE_integer('train_steps', 10000, 'number of training steps')
flags.DEFINE_integer('eval_every', 1000, 'number of steps between evals')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_integer('beam_size', 4, '')
flags.DEFINE_bool('great_transformer', False, 'use great transformer for logic programming?')


flags.DEFINE_integer('gpu', -1, '')
flags.DEFINE_string('port', '29500', 'port for distributed training')
flags.DEFINE_string('gpu_list', None, 'list of gpu ids to be used')
flags.DEFINE_integer('num_proc', 1, '# processes for training')


def get_torch_device(device_id):
  if device_id >= 0 and torch.cuda.is_available():
    return 'cuda:{}'.format(device_id)
  return 'cpu'
