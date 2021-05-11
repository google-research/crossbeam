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
from argparse import Namespace
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
from crossbeam.model.logic_model import LogicModel
from crossbeam.datasets.logic_data_generator import get_consts_and_ops
from crossbeam.datasets.data_gen import trace_gen
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import main_train_eval

FLAGS = flags.FLAGS

def main(argv):
  del argv
  set_global_seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  
  model = LogicModel(FLAGS, operations)

  with open(os.path.join(FLAGS.data_folder, 'test-tasks.pkl'), 'rb') as f:
    eval_tasks = cp.load(f)

  def task_gen(*stuff,**dont_care):
    return random.choice(eval_tasks)
  
  proc_args = Namespace(**FLAGS.flag_values_dict())
  if False: #visualize some execution traces
    for t in eval_tasks:
      tr = list(trace_gen(t.solution))
      print(t)
      for e in tr:
        #print(e.__class__,e)
        print(e.expression())
      print()
      print()
      assert False
  main_train_eval(proc_args, model, eval_tasks, operations, constants, task_gen, trace_gen)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)
