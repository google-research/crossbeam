import torch
import random
import os
import pickle as cp
import glob
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial


def raw_collate_fn(list_tasks):
  list_t, list_inputs_dict, list_outputs, expr_list = zip(*list_tasks)
  return list_t, list_inputs_dict, list_outputs, expr_list


class RawInftyDataset(IterableDataset):
  def __init__(self, seed, task_gen_func, domain, prog_tokenizer=None):
    super(RawInftyDataset, self).__init__()
    if prog_tokenizer is None:
      self.prog_tokenizer = lambda x: x
    else:
      self.prog_tokenizer = prog_tokenizer
    self.seed = seed    
    self.fn_data_gen = task_gen_func
    self.domain = domain

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      random.seed(worker_info.id * 10 + self.seed)

    while True:
      t = self.fn_data_gen(self.domain)
      yield t, t.inputs_dict, t.outputs, self.prog_tokenizer(t.solution.tokenized_expression())

  def collate_fn(self, list_tasks):
    return list_tasks


class RawOfflineDataset(Dataset):
  def __init__(self, pkl_name, prog_tokenizer=None, verbose=True):
    super(RawOfflineDataset, self).__init__()
    if prog_tokenizer is None:
      self.prog_tokenizer = lambda x: x
    else:
      self.prog_tokenizer = prog_tokenizer
    with open(pkl_name, 'rb') as f:
      self.tasks = cp.load(f)
    if verbose:
      print('# loaded for %s' % pkl_name, len(self.tasks))
  
  def __len__(self):
    return len(self.tasks)

  def __getitem__(self, index):
    t = self.tasks[index]
    if t.solution is None:
      sol = []
    else:
      sol = self.prog_tokenizer(t.solution.tokenized_expression())
    return t, t.inputs_dict, t.outputs, sol


def sharded_iterator(train_data_glob, batch_size, prog_tokenizer=None, collate_fn=raw_collate_fn, drop_last=True):
  train_files = sorted(glob.glob(train_data_glob))
  while True:
    for fname in train_files:
      db = RawOfflineDataset(fname, prog_tokenizer, verbose=False)
      idx = list(range(len(db)))
      random.shuffle(idx)
      for i in range(0, len(idx), batch_size):
        if i + batch_size > len(idx) and drop_last:
          break
        lst = [db[j] for j in idx[i : i + batch_size]]
        yield collate_fn(lst)
