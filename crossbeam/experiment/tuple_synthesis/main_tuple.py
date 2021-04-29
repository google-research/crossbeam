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
from absl import logging
from tqdm import tqdm
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from crossbeam.algorithm.synthesis import synthesize
from crossbeam.model.util import CharacterTable
from crossbeam.model.joint_model import JointModel
from crossbeam.datasets.tuple_data_gen import get_consts_and_ops, task_gen, trace_gen
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import train_eval_loop, train_mp
from crossbeam.common.config import get_torch_device
FLAGS = flags.FLAGS


def init_model(operations):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,', max_len=70)
  model = JointModel(FLAGS, input_table, output_table, value_table, operations)  
  return model


def main(argv):
  del argv
  set_global_seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  model = init_model(operations)

  with open(os.path.join(FLAGS.data_folder, 'valid-tasks.pkl'), 'rb') as f:
    eval_tasks = cp.load(f)
  
  proc_args = Namespace(**FLAGS.flag_values_dict())
  if FLAGS.num_proc > 1:
    model.share_memory()
    if FLAGS.gpu_list is not None:
      devices = [get_torch_device(int(x.strip())) for x in FLAGS.gpu_list.split(',')]
    else:
      devices = ['cpu'] * FLAGS.num_proc
    assert len(devices) == FLAGS.num_proc
    nq_per_proc = math.ceil(len(eval_tasks) / FLAGS.num_proc)
    procs = []
    for rank, device in enumerate(devices):
      local_eval_tasks = eval_tasks[rank * nq_per_proc : (rank + 1) * nq_per_proc]
      proc = mp.Process(target=train_eval_loop, args=(proc_args, device, model, 
                                                      local_eval_tasks, 
                                                      operations, constants, task_gen, trace_gen))
      procs.append(proc)
      proc.start()
    for proc in procs:
      proc.join()
  else:
    train_eval_loop(proc_args, get_torch_device(FLAGS.gpu), model, eval_tasks, operations, constants, task_gen, trace_gen)
  logging.info("Training finished!!")


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)
