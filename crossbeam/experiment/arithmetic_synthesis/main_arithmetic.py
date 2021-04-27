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

import random
import numpy as np
import os
import pickle as cp
from absl import app
from absl import flags
from tqdm import tqdm
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossbeam.algorithm.synthesis import synthesize
from crossbeam.model.util import CharacterTable
from crossbeam.model.joint_model import JointModel
from crossbeam.datasets.arithmetic_data_gen import get_consts_and_ops, task_gen, trace_gen
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import singleproc_train_eval_loop

FLAGS = flags.FLAGS


def init_model(operations):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,-', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,-', max_len=70)
  model = JointModel(FLAGS, input_table, output_table, value_table, operations)
  if FLAGS.gpu >= 0:
    model = model.cuda()
    device = 'cuda:{}'.format(FLAGS.gpu)
    model.set_device(device)
  return model


def main(argv):
  del argv
  set_global_seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  model = init_model(operations)
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
  with open(os.path.join(FLAGS.data_folder, 'valid-tasks.pkl'), 'rb') as f:
    eval_tasks = cp.load(f)

  singleproc_train_eval_loop(FLAGS, model, optimizer, eval_tasks, operations, constants, task_gen, trace_gen)


if __name__ == '__main__':
  app.run(main)