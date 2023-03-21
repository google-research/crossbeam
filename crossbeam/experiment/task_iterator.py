import pickle as cp
import numpy as np


class TrainTaskGen(object):
  def __init__(self, weighted_files, local_batch_size, fn_taskgen=None):
    self.weighted_files = weighted_files
    self.local_batch_size = local_batch_size
    self.taskgen = fn_taskgen

  def gen_single_weight(self, rng, weight):
    while True:
      fnames = self.weighted_files[weight][:]
      rng.shuffle(fnames)
      for fname in fnames:
        with open(fname, 'rb') as f:
          list_tasks = cp.load(f)
        rng.shuffle(list_tasks)
        for task in list_tasks:
          yield task

  def datagen(self, seed, probs_of_weights):
    if self.taskgen:
      while True:
        yield [self.taskgen() for _ in range(self.local_batch_size)]

    dict_datagen = {}
    keys = list(probs_of_weights.keys())
    used_keys = []
    probs = []
    for i, key in enumerate(keys):
      rng = np.random.default_rng(seed + i + 1)
      if not key in self.weighted_files:
        continue
      if len(self.weighted_files[key]) == 0:
        continue
      if probs_of_weights[key] > 0:
        dict_datagen[key] = self.gen_single_weight(rng, key)
        probs.append(probs_of_weights[key])
        used_keys.append(key)
    keys = used_keys
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    rng = np.random.default_rng(seed)
    while True:
      batch_tasks = []
      for _ in range(self.local_batch_size):
        weight_idx = np.argmax(rng.multinomial(1, probs))
        weight = keys[weight_idx]
        batch_tasks.append(next(dict_datagen[weight]))
      yield batch_tasks


class EvalTaskGen(TrainTaskGen):
  def __init__(self, num_local_eval, weighted_files):
    super().__init__(weighted_files, 1, None)
    self.num_local_eval = num_local_eval

  def datagen(self, seed, probs_of_weights):
    generator = super().datagen(seed, probs_of_weights)
    for _ in range(self.num_local_eval):
      yield next(generator)[0]


class TaskScheduler(object):
  def __init__(self, args, all_weights, override_schedule_type=None):
    self.all_weights = sorted(list(all_weights))
    self.schedule_type = args.get('schedule_type', 'uniform')
    if override_schedule_type is not None:
      self.schedule_type = override_schedule_type
    self.steps_per_curr_stage = args.get('steps_per_curr_stage', 0)

  def get_schedule(self, step):
    probs_of_weights = {}
    if self.schedule_type == 'uniform':
      for key in self.all_weights:
        probs_of_weights[key] = 1.0 / len(self.all_weights)
    elif self.schedule_type.startswith('halfhalf'):
      if '-' in self.schedule_type:
        stage = int(self.schedule_type.split('-')[1])
      else:
        stage = 0
      assert self.steps_per_curr_stage > 0
      stage += step // self.steps_per_curr_stage
      stage = min(stage, len(self.all_weights) - 1)
      if stage == 0:
        probs_of_weights[self.all_weights[0]] = 1.0
      else:
        probs_of_weights[self.all_weights[stage]] = 0.5
        for s in range(stage):
          probs_of_weights[self.all_weights[s]] = 0.5 / stage
    elif self.schedule_type.startswith('all-'):
      idx = int(self.schedule_type.split('-')[1])
      w = 1.0 / (idx + 1)
      for i in range(idx + 1):
        probs_of_weights[self.all_weights[i]] = w
    else:
      raise NotImplementedError
    return probs_of_weights
