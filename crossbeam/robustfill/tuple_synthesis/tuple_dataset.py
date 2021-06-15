import torch
import random
import os
import pickle as cp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial


def raw_collate_fn(list_tasks):
  list_inputs_dict, list_outputs, expr_list = zip(*list_tasks)
  return list_inputs_dict, list_outputs, expr_list


class RawTupleInftyDataset(IterableDataset):
  def __init__(self, seed, task_gen_func, domain):
    super(RawTupleInftyDataset, self).__init__()
    self.seed = seed    
    self.fn_data_gen = task_gen_func
    self.domain = domain

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      random.seed(worker_info.id * 10 + self.seed)

    while True:
      t = self.fn_data_gen(self.domain)
      yield t.inputs_dict, t.outputs, t.solution.tokenized_expression()

  def collate_fn(self, list_tasks):
    return list_tasks


class RawTupleOfflineDataset(Dataset):
  def __init__(self, data_folder, phase):
    super(RawTupleOfflineDataset, self).__init__()  
    with open(os.path.join(data_folder, '%s-tasks.pkl' % phase), 'rb') as f:
      self.tasks = cp.load(f)
    print('# %s tasks' % phase, len(self.tasks))
  
  def __len__(self):
    return len(self.tasks)

  def __getitem__(self, index):
    t = self.tasks[index]
    return t.inputs_dict, t.outputs, t.solution.tokenized_expression()
