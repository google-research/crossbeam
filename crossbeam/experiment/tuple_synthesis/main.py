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
FLAGS = flags.FLAGS

from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module
from crossbeam.algorithm.synthesis import synthesize
from crossbeam.model.util import CharacterTable
from crossbeam.model.op_arg import LSTMArgSelector
from crossbeam.model.op_init import OpPoolingState
from crossbeam.model.encoder import CharIOLSTMEncoder, CharValueLSTMEncoder

from crossbeam.datasets.tuple_data_gen import get_consts_and_ops, task_gen, trace_gen

# nn configs
flags.DEFINE_integer('embed_dim', 128, 'embedding dimension')
flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_boolean('score_normed', True, 'whether to normalize the score into valid probability')

flags.DEFINE_integer('gpu', -1, '')
flags.DEFINE_integer('beam_size', 4, '')
flags.DEFINE_float('grad_clip', 5.0, 'clip grad')
flags.DEFINE_integer('max_search_weight', 8, '')

flags.DEFINE_integer('train_steps', 10000, 'number of training steps')
flags.DEFINE_integer('eval_every', 1000, 'number of steps between evals')
flags.DEFINE_float('lr', 0.0001, 'learning rate')


class JointModel(nn.Module):
  def __init__(self, input_table, output_table, value_table, operations):
    super(JointModel, self).__init__()
    self.device = 'cpu'
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=FLAGS.embed_dim)
    self.val = CharValueLSTMEncoder(value_table, hidden_size=FLAGS.embed_dim)
    self.arg = LSTMArgSelector(hidden_size=FLAGS.embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=FLAGS.step_score_func,
                               step_score_normalize=FLAGS.score_normed)
    self.init = OpPoolingState(ops=tuple(operations), state_dim=FLAGS.embed_dim, pool_method='mean')

  def set_device(self, device):
    self.device = device
    self.io.set_device(device)
    self.val.set_device(device)
    self.arg.set_device(device)


def init_model(operations):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,', max_len=70)
  model = JointModel(input_table, output_table, value_table, operations)
  if FLAGS.gpu >= 0:
    model = model.cuda()
    device = 'cuda:{}'.format(FLAGS.gpu)
    model.set_device(device)
  return model


def train_step(task, training_samples, all_values, model, optimizer):
  optimizer.zero_grad()
  io_embed = model.io(task.inputs_dict, task.outputs)
  val_embed = model.val(all_values)
  loss = 0.0
  for sample in training_samples:
    arg_options, true_arg_pos, num_vals, op = sample
    arg_options = torch.LongTensor(arg_options).to(model.device)
    cur_vals = val_embed[:num_vals]
    op_state = model.init(io_embed, cur_vals, op)
    scores = model.arg(op_state, cur_vals, arg_options)
    scores = torch.sum(scores, dim=-1)
    if FLAGS.score_normed:
        nll = -scores[true_arg_pos]
    else:
        nll = -F.log_softmax(scores, dim=0)[true_arg_pos]
    loss = loss + nll
  loss = loss / len(training_samples)
  loss.backward()
  if FLAGS.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=FLAGS.grad_clip)
  optimizer.step()
  return loss


def do_eval(eval_tasks, operations, constants, model):
  print('doing eval...')
  succ = 0.0
  for t in eval_tasks:
    out, _ = synthesize(t, operations, constants, model,
                        max_weight=FLAGS.max_search_weight,
                        k=FLAGS.beam_size,
                        is_training=False)
    if out is not None:
      succ += 1.0
  succ /= len(eval_tasks)
  print('eval success rate: {:.1f}%'.format(succ * 100))
  return succ


def main(argv):
  del argv
  torch.manual_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  model = init_model(operations)
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
  with open(os.path.join(FLAGS.data_folder, 'valid-tasks.pkl'), 'rb') as f:
    eval_tasks = cp.load(f)
  pbar = tqdm(range(FLAGS.train_steps))
  for i in pbar:
    if i % FLAGS.eval_every == 0:
      do_eval(eval_tasks, operations, constants, model)
    t = task_gen(FLAGS, constants, operations)
    trace = list(trace_gen(t.solution))
    with torch.no_grad():
      training_samples, all_values = synthesize(t, operations, constants, model,
                                                trace=trace,
                                                max_weight=FLAGS.max_search_weight,
                                                k=FLAGS.beam_size,
                                                is_training=True)
    if isinstance(training_samples, list):
      loss = train_step(t, training_samples, all_values, model, optimizer)
      pbar.set_description('train loss: %.2f' % loss)

  print('Training finished. Performing final evaluation...')
  do_eval(eval_tasks, operations, constants, model)


if __name__ == '__main__':
  app.run(main)
