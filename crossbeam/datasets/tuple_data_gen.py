from crossbeam.common import config

import random
import numpy as np
from absl import app
from absl import flags
import pickle as cp
from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module

FLAGS = flags.FLAGS

flags.DEFINE_string('output_file', None, 'data dump')
flags.DEFINE_integer('num_eval', 100, '# tasks for evaluation')
flags.DEFINE_integer('num_examples', 2, '')
flags.DEFINE_integer('num_inputs', 3, '')
flags.DEFINE_integer('min_task_weight', 3, '')
flags.DEFINE_integer('max_task_weight', 6, '')


def task_gen(args, constants, operations):
  return random_data.generate_good_random_task(
      min_weight=args.min_task_weight,
      max_weight=args.max_task_weight,
      num_examples=args.num_examples,
      num_inputs=args.num_inputs,
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


def get_consts_and_ops():
  operations = tuple_operations.get_operations()
  constants = [0]
  return constants, operations


def main(argv):
  del argv
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  eval_tasks = [task_gen(FLAGS, constants, operations) for _ in range(FLAGS.num_eval)]

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
