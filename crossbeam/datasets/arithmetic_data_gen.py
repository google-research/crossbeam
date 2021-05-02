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

from crossbeam.datasets.tuple_data_gen import task_gen

FLAGS = flags.FLAGS


def get_consts_and_ops():
  operations = arithmetic_operations.get_operations()
  constants = [0]
  return constants, operations


def trace_gen(value_node):
  if isinstance(value_node, value_module.OperationValue): # non-leaf
    sub_ops = []
    for ch in value_node.arg_values:
      if isinstance(ch, value_module.OperationValue): # non-leaf
        sub_ops.append(ch)
    random.shuffle(sub_ops)
    for ch in sub_ops:
      sub_trace = trace_gen(ch)
      for v in sub_trace:
        yield v
    yield value_node


def main(argv):
  del argv
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  eval_tasks = [task_gen(FLAGS, constants, operations) for _ in range(FLAGS.num_eval)]
  import pdb; pdb.set_trace()
  

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
