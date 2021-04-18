import random
import numpy as np
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


flags.DEFINE_integer('seed', 1, 'random seed')
# nn configs
flags.DEFINE_integer('embed_dim', 128, 'embedding dimension')
flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_boolean('score_normed', False, 'whether to normalize the score into valid probability')

flags.DEFINE_integer('gpu', -1, '')
flags.DEFINE_integer('beam_size', 4, '')
flags.DEFINE_integer('max_search_weight', 8, '')

flags.DEFINE_integer('num_eval', 10, '')

flags.DEFINE_integer('num_examples', 2, '')
flags.DEFINE_integer('num_inputs', 3, '')
flags.DEFINE_integer('min_task_weight', 3, '')
flags.DEFINE_integer('max_task_weight', 6, '')


class JointModel(nn.Module):
  def __init__(self, input_table, output_table, value_table, operations):
    super(JointModel, self).__init__()
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=FLAGS.embed_dim)
    self.val = CharValueLSTMEncoder(value_table, hidden_size=FLAGS.embed_dim)
    self.arg = LSTMArgSelector(hidden_size=FLAGS.embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=FLAGS.step_score_func,
                               step_score_normalize=FLAGS.score_normed)
    self.init = OpPoolingState(ops=tuple(operations), state_dim=FLAGS.embed_dim, pool_method='mean')


def init_model(operations):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,', max_len=70)
  model = JointModel(input_table, output_table, value_table, operations)
  if FLAGS.gpu >= 0:
    model = model.cuda()
  return model


def task_gen(constants, operations):
  while True:
    task = random_data.generate_random_task(
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        num_examples=FLAGS.num_examples,
        num_inputs=FLAGS.num_inputs,
        constants=constants,
        operations=operations,
        input_generator=random_data.RANDOM_INTEGER)  
    if task:
      return task


def trace_gen(value_node):
  if isinstance(value_node, value_module.OperationValue): # non-leaf
    for value in value_node.arg_values:
      sub_trace = trace_gen(value)
      for v in sub_trace:
        yield v
    yield value_node


def train_step(task, training_samples, all_values, model, optimizer):
  optimizer.zero_grad()
  io_embed = model.io(task.inputs_dict, task.outputs)
  val_embed = model.val(all_values)
  loss = 0.0
  for sample in training_samples:
    arg_options, true_arg_pos, num_vals, op = sample
    arg_options = torch.LongTensor(arg_options)
    cur_vals = val_embed[:num_vals]
    op_state = model.init(io_embed, cur_vals, op)
    scores = model.arg(op_state, cur_vals, arg_options)
    scores = torch.sum(scores, dim=-1)
    nll = -F.log_softmax(scores, dim=0)[true_arg_pos]
    loss = loss + nll
  loss = loss / len(training_samples)
  loss.backward()
  optimizer.step()
  return loss


def do_eval(eval_tasks, operations, constants, model):
  print('doing eval')
  succ = 0.0
  for t in tqdm(eval_tasks):
    out, _ = synthesize(t, operations, constants, model, 
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
  torch.manual_seed(3)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  operations = tuple_operations.get_operations()
  constants = [0]
  model = init_model(operations)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  eval_tasks = [task_gen(constants, operations) for _ in range(FLAGS.num_eval)]
  do_eval(eval_tasks, operations, constants, model)
  pbar = tqdm(range(1000))
  for i in pbar:
    # t = task_gen(constants, operations)
    t = eval_tasks[i % len(eval_tasks)]
    trace = list(trace_gen(t.solution))
    with torch.no_grad():
      training_samples, all_values = synthesize(t, operations, constants, model,
                                                trace=trace,
                                                max_weight=FLAGS.max_search_weight,
                                                k=FLAGS.beam_size,
                                                is_training=True)
    if isinstance(training_samples, list):
      loss = train_step(t, training_samples, all_values, model, optimizer)
      pbar.set_description('loss: %.2f' % loss)
  do_eval(eval_tasks, operations, constants, model)


if __name__ == '__main__':
  app.run(main)
