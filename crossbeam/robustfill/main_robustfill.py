import random
import os
import glob
import numpy as np
import sys
from absl import app
from absl import flags
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import functools

from torch.utils.data import DataLoader
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.dsl import checker
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.model.util import CharacterTable
from crossbeam.model.encoder import CharIOLSTMEncoder
from crossbeam.model.op_init import IOPoolProjSummary
from crossbeam.robustfill.rnn_prog import RfillAutoreg
from crossbeam.robustfill.dataset import raw_collate_fn, RawInftyDataset, RawOfflineDataset, sharded_iterator

FLAGS = flags.FLAGS


class RobustFill(nn.Module):
  def __init__(self, input_table, output_table, prog_vdict):
    super(RobustFill, self).__init__()
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=FLAGS.embed_dim)
    self.io_summary = IOPoolProjSummary(FLAGS.embed_dim, pool_method='mean')
    self.decoder = RfillAutoreg(prog_vdict, FLAGS.decoder_rnn_layers, FLAGS.embed_dim)

  def embed_io(self, list_inputs_dict, list_outputs, device):
    io_concat_embed, sample_scatter_idx = self.io(list_inputs_dict, list_outputs, 
                                                  device=device, needs_scatter_idx=True)
    return self.io_summary(io_concat_embed, sample_scatter_idx)


def init_model(domain):
  input_table = CharacterTable(domain.input_charset,
                                max_len=domain.input_max_len)
  output_table = CharacterTable(domain.output_charset,
                                max_len=domain.output_max_len)

  prog_vocab = ['pad'] + [str(x) for x in domain.constants]
  for i in range(1, FLAGS.max_num_inputs + 1):
      prog_vocab.append('in%d' % i)
  prog_vocab += domain.program_tokens
  prog_vocab += ['sos', 'eos']
  prog_vdict = {}
  for i, v in enumerate(prog_vocab):
    prog_vdict[v] = i
  model = RobustFill(input_table, output_table, prog_vdict)
  return model


def eval_dataset(model, dataset, device):
  eval_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, collate_fn=raw_collate_fn, num_workers=0,
                            shuffle=False, drop_last=False)
  model.eval()
  with torch.no_grad():
    hit_at_1 = 0.0
    total_hit = 0.0
    for list_tasks, list_inputs_dict, list_outputs, expr_list in eval_loader:
      init_state = model.embed_io(list_inputs_dict, list_outputs, device=device)
      _, list_pred_progs, _, _ = model.decoder.beam_search(init_state, beam_size=FLAGS.beam_size, max_len=50)
      for i in range(len(list_tasks)):
        hit = -1
        for j in range(FLAGS.beam_size):
          pred_prog = ''.join(list_pred_progs[i * FLAGS.beam_size + j])
          if checker.check_solution(list_tasks[i], pred_prog):
            hit = j
            break
          assert pred_prog != ''.join(expr_list[i])
        if hit >= 0:
          hit_at_1 += hit == 0
          total_hit += 1
    hit_at_1 = hit_at_1 * 100.0 / len(dataset)
    total_hit = total_hit * 100.0 / len(dataset)
  return hit_at_1, total_hit


def main(argv):
  del argv
  set_global_seed(FLAGS.seed)
  domain = domains.get_domain(FLAGS.domain)
  task_gen_func = functools.partial(
      data_gen.task_gen,
      min_weight=FLAGS.min_task_weight,
      max_weight=FLAGS.max_task_weight,
      min_num_examples=FLAGS.min_num_examples,
      max_num_examples=FLAGS.max_num_examples,
      min_num_inputs=FLAGS.min_num_inputs,
      max_num_inputs=FLAGS.max_num_inputs,
      verbose=FLAGS.verbose)

  valid_dataset = RawOfflineDataset(glob.glob(os.path.join(FLAGS.data_folder, 'valid-tasks'))[0])
  if FLAGS.train_data_glob is None:
    train_dataset = RawInftyDataset(FLAGS.seed, task_gen_func, domain)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              collate_fn=raw_collate_fn, num_workers=FLAGS.n_para_dataload)
    train_gen = iter(train_loader)
  else:
    train_gen = sharded_iterator(os.path.join(FLAGS.data_folder, FLAGS.train_data_glob),
                                 batch_size=FLAGS.batch_size)

  model = init_model(domain)
  if FLAGS.load_model is not None:
   model_dump = os.path.join(FLAGS.save_dir, FLAGS.load_model)
   print('loading from', model_dump)
   model.load_state_dict(torch.load(model_dump))
  if FLAGS.gpu >= 0:
    model = model.cuda()
    device = 'cuda:{}'.format(FLAGS.gpu)
  else:
    device = 'cpu'    
  if FLAGS.do_test:
    test_dataset = RawOfflineDataset(glob.glob(os.path.join(FLAGS.data_folder, 'test-tasks'))[0])
    hit_at_1, total_hit = eval_dataset(model, test_dataset, device)
    print('test hit@1: %.2f, hit@%d: %.2f' % (hit_at_1, FLAGS.beam_size, total_hit))
    sys.exit()
  optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

  cur_step = 0
  best_valid = -1
  while cur_step < FLAGS.train_steps:
    pbar = tqdm(range(cur_step, cur_step + FLAGS.eval_every))
    model.train()
    for it in pbar:
      _, list_inputs_dict, list_outputs, expr_list = next(train_gen)
      optimizer.zero_grad()
      init_state = model.embed_io(list_inputs_dict, list_outputs, device=device)
      ll = model.decoder(init_state, expr_list)
      loss = -torch.mean(ll)
      loss.backward()
      if FLAGS.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=FLAGS.grad_clip)
      optimizer.step()
      pbar.set_description('iter: %d, loss: %.4f' % (it, loss.item()))
    cur_step += FLAGS.eval_every
    hit_at_1, total_hit = eval_dataset(model, valid_dataset, device)
    print('valid hit@1: %.2f, hit@%d: %.2f' % (hit_at_1, FLAGS.beam_size, total_hit))
    if hit_at_1 > best_valid:
      print('saving best valid model')
      best_valid = hit_at_1
      save_file = os.path.join(FLAGS.save_dir, 'model-best-valid.ckpt')
      torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    app.run(main)
