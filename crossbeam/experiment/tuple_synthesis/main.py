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
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from tqdm import tqdm
from flax import optim
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
import functools
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


flags.DEFINE_integer('seed', 1, 'random seed')
# nn configs
flags.DEFINE_integer('embed_dim', 128, 'embedding dimension')
flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_boolean('score_normed', False, 'whether to normalize the score into valid probability')


flags.DEFINE_integer('beam_size', 4, '')
flags.DEFINE_integer('max_search_weight', 8, '')

flags.DEFINE_integer('num_eval', 100, '')
flags.DEFINE_integer('train_steps', 10000, '')

flags.DEFINE_integer('num_examples', 2, '')
flags.DEFINE_integer('num_inputs', 3, '')
flags.DEFINE_integer('min_task_weight', 3, '')
flags.DEFINE_integer('max_task_weight', 6, '')


def init_model(key, operations):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,-', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,-', max_len=70)

  model = {}
  model['io'] = CharIOLSTMEncoder(input_table, output_table, hidden_size=FLAGS.embed_dim)
  model['val'] = CharValueLSTMEncoder(value_table, hidden_size=FLAGS.embed_dim)
  model['arg'] = LSTMArgSelector(hidden_size=FLAGS.embed_dim,
                                 step_score_func=FLAGS.step_score_func,
                                 step_score_normalize=FLAGS.score_normed)
  model['init'] = OpPoolingState(ops=tuple(operations), state_dim=FLAGS.embed_dim)
  model = FrozenDict(**model)
  rand_keys = jax.random.split(key, num=len(model))
  params = {}
  for i, name in enumerate(model):
    mod = model[name]
    params[name] = mod.init_params(rand_keys[i])
  return model, params


def task_gen(constants, operations):
  return random_data.generate_good_random_task(
      min_weight=FLAGS.min_task_weight,
      max_weight=FLAGS.max_task_weight,
      num_examples=FLAGS.num_examples,
      num_inputs=FLAGS.num_inputs,
      constants=constants,
      operations=operations,
      input_generator=random_data.RANDOM_INTEGER)


def trace_gen(value_node):
  if isinstance(value_node, value_module.OperationValue): # non-leaf
    for value in value_node.arg_values:
      sub_trace = trace_gen(value)
      for v in sub_trace:
        yield v
    yield value_node


@functools.partial(jax.jit, static_argnums=[0, 8])
def single_fwd_backwd(model, optimizer, io_input, val_input, val_mask, padded_args, arg_mask, true_label, op):
  def loss(params):
    io_embed = model['io'].exec_encode(params['io'], *io_input)
    val_embed = model['val'].exec_encode(params['val'], *val_input)
    op_state = model['init'].encode(params['init'], io_embed, val_embed, val_mask, op)
    scores = model['arg'].apply(params['arg'], op_state, val_embed, padded_args).flatten()
    scores = scores * arg_mask + (1 - arg_mask) * -1e10
    nll = -nn.log_softmax(scores)[true_label]
    return nll
  l, grads = jax.value_and_grad(loss)(optimizer.target)
  optimizer = optimizer.apply_gradient(grads)
  return l, optimizer


def train_step(task, training_samples, all_values, model, optimizer):
  val_input = model['val'].make_input(all_values)
  io_input = model['io'].make_input(task.inputs_dict, task.outputs)

  loss = 0.0
  for sample in training_samples:
    arg_options, true_arg_pos, num_vals, op = sample
    true_arg_pos = jnp.array(true_arg_pos, dtype=jnp.int32)
    arg_options = jnp.asarray(arg_options)
    pad_num = FLAGS.beam_size + 1 - arg_options.shape[0]
    arg_mask = jnp.pad(jnp.ones((arg_options.shape[0],), dtype=jnp.float32), [(0, pad_num)])
    val_mask = jnp.pad(jnp.ones((num_vals,), dtype=jnp.float32), [(0, val_input[0].shape[0] - num_vals)])
    padded_args = jnp.pad(arg_options, [(0, pad_num), (0, 0)])
    l, optimizer = single_fwd_backwd(model, optimizer, io_input, val_input, val_mask, padded_args, arg_mask, true_arg_pos, op)
    loss = loss + l.item()

  return loss / len(training_samples), optimizer


def do_eval(eval_tasks, operations, constants, model, params):
  print('doing eval')
  succ = 0.0
  for t in tqdm(eval_tasks):
    out, _ = synthesize(t, operations, constants, model, params,
                        max_weight=FLAGS.max_search_weight,
                        k=FLAGS.beam_size,
                        is_training=False)
    if out is not None:
      succ += 1.0
  succ /= len(eval_tasks)
  print('eval succ:', succ)
  return succ


def main(argv):
  del argv
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  key = jax.random.PRNGKey(FLAGS.seed)

  operations = tuple_operations.get_operations()
  constants = [0]
  model, params = init_model(key, operations)
  optimizer_def = optim.Adam(0.0001)
  optimizer = optimizer_def.create(params)

  eval_tasks = [task_gen(constants, operations) for _ in range(FLAGS.num_eval)]
  do_eval(eval_tasks, operations, constants, model, optimizer.target)
  pbar = tqdm(range(FLAGS.train_steps))
  for i in pbar:
    # t = task_gen(constants, operations)
    t = eval_tasks[i % len(eval_tasks)]
    trace = list(trace_gen(t.solution))
    training_samples, all_values = synthesize(t, operations, constants, model, optimizer.target,
                                              trace=trace,
                                              max_weight=FLAGS.max_search_weight,
                                              k=FLAGS.beam_size,
                                              is_training=True)
    if isinstance(training_samples, list):
      loss, optimizer = train_step(t, training_samples, all_values, model, optimizer)
      pbar.set_description('loss: %.2f' % loss)
  do_eval(eval_tasks, operations, constants, model, optimizer.target)


if __name__ == '__main__':
  app.run(main)
