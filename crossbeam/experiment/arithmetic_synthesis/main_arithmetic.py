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
from argparse import Namespace
from absl import app
from absl import flags
from tqdm import tqdm
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossbeam.algorithm.synthesis import synthesize
from crossbeam.model.util import CharacterTable
from crossbeam.model.joint_model import JointModel, IntJointModel
from crossbeam.datasets.data_gen import arithmetic_consts_and_ops, task_gen, trace_gen
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import main_train_eval
FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'char', 'int/char')

def init_model(operations):
  if FLAGS.model_type == 'char':
    input_table = CharacterTable('0123456789:,', max_len=50)
    output_table = CharacterTable('0123456789() ,-', max_len=50)
    value_table = CharacterTable('0123456789intuple:[]() ,-', max_len=70)
    model = JointModel(FLAGS, input_table, output_table, value_table, operations)
  elif FLAGS.model_type == 'int':
    model = IntJointModel(FLAGS, 
                          input_range=(0, 10), 
                          output_range=(-800, 800),
                          value_range=(-800, 800),
                          operations=operations)
  else:
    raise ValueError('unknown model type %s' % FLAGS.model_type)
  return model


def main(argv):
  del argv
  set_global_seed(FLAGS.seed)

  constants, operations = arithmetic_consts_and_ops()
  model = init_model(operations)

  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
  with open(os.path.join(FLAGS.data_folder, 'valid-tasks.pkl'), 'rb') as f:
    eval_tasks = cp.load(f)

  proc_args = Namespace(**FLAGS.flag_values_dict())
  main_train_eval(proc_args, model, eval_tasks, operations, constants, task_gen, trace_gen)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)
