from absl import flags
import torch
import random
import numpy as np

flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_string('train_data_glob', None, 'offline data dumps')
flags.DEFINE_boolean('op_in_beam', False, 'op selection as part of beam search?')
flags.DEFINE_boolean('batch_training', False, 'do batch training?')
flags.DEFINE_boolean('use_ur', True, 'use UR for evaluation?')
flags.DEFINE_boolean('score_normed', True, 'whether to normalize the score into valid probability')
flags.DEFINE_integer('grad_accumulate', 1, '# forward / backward steps')
flags.DEFINE_integer('max_search_weight', 12, '')
flags.DEFINE_integer('num_valid', -1, 'num tasks for evaluation per process during training')
flags.DEFINE_float('timeout', 5, 'time limit in seconds')


def set_global_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
