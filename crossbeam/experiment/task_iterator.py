import pickle as cp
import random
import numpy as np


class TrainTaskGen(object):
  def __init__(self, weighted_files, local_batch_size, fn_taskgen=None):
    self.weighted_files = weighted_files
    self.local_batch_size = local_batch_size
    self.taskgen = fn_taskgen
    assert self.taskgen is None  #TODO(hadai) see if we still need this

  def gen_single_weight(self, weight):
    while True:
      for fname in self.weighted_files[weight]:
        with open(fname, 'rb') as f:
          list_tasks = cp.load(f)
        random.shuffle(list_tasks)
        for task in list_tasks:
          yield task
        for i in range(0, len(list_tasks), self.local_batch_size):
          yield list_tasks[i : i + self.local_batch_size]

  def datagen(self, probs_of_weights):
    dict_datagen = {}
    keys = list(probs_of_weights.keys())
    probs = []
    for key in keys:
      if probs_of_weights[key] > 0 and key in self.weighted_files:
        dict_datagen[key] = self.gen_single_weight(key)
        probs.append(probs_of_weights[key])
    probs = np.array(probs)
    probs = probs / np.sum(probs)

    while True:
      batch_tasks = []
      for _ in range(self.local_batch_size):
        weight_idx = np.argmax(np.random.multinomial(1, probs))
        weight = keys[weight_idx]
        batch_tasks.append(next(dict_datagen[weight]))
      yield batch_tasks


class TaskScheduler(object):
  def __init__(self, args, all_weights):
    self.all_weights = list(all_weights)
    self.schedule_type = 'uniform'

  def get_schedule(self, step):
    if self.schedule_type == 'uniform':
      probs_of_weights = {}
      for key in self.all_weights:
        probs_of_weights[key] = 1.0 / len(self.all_weights)
      return probs_of_weights
    else:
      raise NotImplementedError
