from absl import flags
import torch
import random
import numpy as np

flags.DEFINE_string('pooling', 'mean', 'pooling method used')
flags.DEFINE_string('step_score_func', 'mlp', 'score func used at each step of autoregressive model')
flags.DEFINE_boolean('score_normed', True, 'whether to normalize the score into valid probability')
flags.DEFINE_integer('grad_accumulate', 1, '# forward / backward steps')
flags.DEFINE_integer('max_search_weight', 12, '')


def set_global_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
