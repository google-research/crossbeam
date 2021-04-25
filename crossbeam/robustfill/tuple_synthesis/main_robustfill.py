import random
import os
import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial

from torch.utils.data import DataLoader, Dataset, IterableDataset
from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module
from crossbeam.datasets.tuple_data_gen import get_consts_and_ops
from crossbeam.model.util import CharacterTable
from crossbeam.model.encoder import CharIOLSTMEncoder
from crossbeam.model.op_init import IOPoolProjSummary
from crossbeam.robustfill.rnn_prog import RfillAutoreg
from crossbeam.robustfill.tuple_synthesis.tuple_dataset import raw_collate_fn, RawTupleInftyDataset, RawTupleOfflineDataset

FLAGS = flags.FLAGS


class RobustFill(nn.Module):
  def __init__(self, input_table, output_table, prog_vdict):
    super(RobustFill, self).__init__()
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=FLAGS.embed_dim)
    self.io_summary = IOPoolProjSummary(FLAGS.embed_dim, pool_method='max')
    self.decoder = RfillAutoreg(prog_vdict, FLAGS.decoder_rnn_layers, FLAGS.embed_dim)

  def embed_io(self, list_inputs_dict, list_outputs):
    io_concat_embed, sample_scatter_idx = self.io(list_inputs_dict, list_outputs, needs_scatter_idx=True)
    return self.io_summary(io_concat_embed, sample_scatter_idx)

  def set_device(self, device):
    self.device = device
    self.io.set_device(device)


def init_model(constants):
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,', max_len=50)

  prog_vocab = ['pad'] + [str(x) for x in constants]
  for i in range(1, FLAGS.num_inputs + 1):
      prog_vocab.append('in%d' % i)
  prog_vocab += ['(', ')', ', ', 'sos', 'eos']
  prog_vdict = {}
  for i, v in enumerate(prog_vocab):
    prog_vdict[v] = i
  model = RobustFill(input_table, output_table, prog_vdict)
  return model


def main(argv):
  del argv
  torch.manual_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  constants, operations = get_consts_and_ops()

  train_dataset = RawTupleInftyDataset(FLAGS.seed, FLAGS, constants, operations)
  valid_dataset = RawTupleOfflineDataset(FLAGS.data_folder, 'valid')
  train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, 
                            collate_fn=raw_collate_fn, num_workers=FLAGS.n_para_dataload)
  train_gen = iter(train_loader)

  model = init_model(constants)
  if FLAGS.gpu >= 0:
    model = model.cuda()
    device = 'cuda:{}'.format(FLAGS.gpu)
    model.set_device(device)
  optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)  

  cur_step = 0
  best_valid = -1
  while cur_step < FLAGS.train_steps:
    pbar = tqdm(range(cur_step, cur_step + FLAGS.eval_every))
    model.train()
    for it in pbar:
      list_inputs_dict, list_outputs, expr_list = next(train_gen)
      optimizer.zero_grad()
      init_state = model.embed_io(list_inputs_dict, list_outputs)
      ll = model.decoder(init_state, expr_list)
      loss = -torch.mean(ll)
      loss.backward()
      if FLAGS.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=FLAGS.grad_clip)
      optimizer.step()
      pbar.set_description('iter: %d, loss: %.4f' % (it, loss.item()))
    cur_step += FLAGS.eval_every
    valid_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, collate_fn=raw_collate_fn, num_workers=0,
                              shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
      hit_at_1 = 0.0
      for list_inputs_dict, list_outputs, expr_list in valid_loader:
        init_state = model.embed_io(list_inputs_dict, list_outputs)
        _, list_pred_progs, _, _ = model.decoder.beam_search(init_state, beam_size=FLAGS.beam_size, max_len=50)        
        for i in range(len(expr_list)):
          pred = ''.join(list_pred_progs[i * FLAGS.beam_size])
          if pred == ''.join(expr_list[i]):
            hit_at_1 += 1
      hit_at_1 = hit_at_1 * 100.0 / len(valid_dataset)
      print('valid top1: %.2f' % hit_at_1)
      if hit_at_1 > best_valid:
        print('saving best valid model')
        best_valid = hit_at_1
        save_file = os.path.join(FLAGS.save_dir, 'model-best-valid.ckpt')
        torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    app.run(main)