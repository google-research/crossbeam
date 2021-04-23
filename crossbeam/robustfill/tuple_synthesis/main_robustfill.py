import random
import os
import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm
import pickle as cp
import torch
from functools import partial
from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module

from crossbeam.datasets.tuple_data_gen import task_gen, get_consts_and_ops

FLAGS = flags.FLAGS
flags.DEFINE_string('data_folder', None, 'folder for valid/test data')


def main(argv):
  del argv
  torch.manual_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  constants, operations = get_consts_and_ops()

  with open(os.path.join(FLAGS.data_folder, 'valid-tasks.pkl'), 'rb') as f:
    valid_tasks = cp.load(f)
    print('# valid', len(valid_tasks))
  fn_data_gen = partial(task_gen, FLAGS, constants, operations)
  t = fn_data_gen()



if __name__ == '__main__':
    app.run(main)