from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

from crossbeam.model.base import MLP
from crossbeam.common.consts import N_INF, EPS
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pad_sequence
from torch_scatter import scatter_add


def beam_step(raw_scores, cur_sizes, beam_size):
  pad_size = max(cur_sizes)
  batch_size = len(cur_sizes)
  n_choices = raw_scores.shape[1]
  if pad_size != min(cur_sizes):
    raw_scores = raw_scores.split(cur_sizes, dim=0)
    padded_scores = pad_sequence(raw_scores, batch_first=True, padding_value=N_INF)
  else:
    padded_scores = raw_scores

  padded_scores = padded_scores.view(batch_size, -1)
  topk_scores, candidates = padded_scores.topk(min(beam_size, padded_scores.shape[1]), dim=-1, sorted=True)
  pred_opts = candidates % n_choices
  pos_index = []
  gap = 0
  for i, s in enumerate(cur_sizes):
    pos_index.append(i * pad_size - gap)
    gap += pad_size - s
  pos_index = torch.LongTensor(pos_index).to(padded_scores.device).view(-1, 1)
  predecessors = candidates // n_choices + pos_index.expand_as(candidates)
  valid = topk_scores > EPS + N_INF
  n_valid = valid.sum(dim=-1)
  cur_sizes = n_valid.data.cpu().numpy().tolist()
  predecessors = predecessors[valid]
  pred_opts = pred_opts[valid]
  scores = topk_scores[valid].view(-1, 1)
  return predecessors, pred_opts, scores, cur_sizes


class RfillAutoreg(nn.Module):
  def __init__(self, vocab, rnn_layers, embed_dim):
    super(RfillAutoreg, self).__init__()
    self.cell_type = 'lstm'
    self.vocab = vocab
    self.tok_start = self.vocab['sos']
    self.tok_stop = self.vocab['eos']
    self.tok_pad = self.vocab['pad']
    decision_mask = torch.ones(2, len(self.vocab))
    decision_mask[1] = 0.0
    decision_mask[1, self.tok_stop] = 1.0
    self.register_buffer('decision_mask', decision_mask)

    state_trans = torch.zeros(2, len(self.vocab), dtype=torch.int64)
    state_trans[0, self.tok_stop] = 1
    state_trans[1] = 1
    self.register_buffer('state_trans', state_trans)

    assert self.tok_pad == 0
    self.inv_map = {}
    for key in self.vocab:
        self.inv_map[self.vocab[key]] = key
    self.tok_embed = nn.Embedding(len(self.vocab), embed_dim)
    self.rnn = nn.LSTM(embed_dim, embed_dim, rnn_layers, bidirectional=False)
    self.out_pred = nn.Linear(embed_dim, len(self.vocab))

  def _get_onehot(self, idx, vsize):
    out_shape = idx.shape + (vsize,)
    idx = idx.view(-1, 1)
    out = torch.zeros(idx.shape[0], vsize).to(idx.device)
    out.scatter_(1, idx, 1)
    return out.view(out_shape)

  def get_init_state(self, state):
    if not isinstance(state, tuple):
      _, state = self.rnn(state.unsqueeze(0))
    return state

  def prog2idx(self, expr):
    return torch.LongTensor([self.tok_start] + [self.vocab[c] for c in expr] + [self.tok_stop])

  def get_likelihood(self, state, expr_list, enforce_sorted, cooked_data=None):
    int_seqs = [self.prog2idx(x).to(state[0].device) for x in expr_list]
    packed_seq = pack_sequence(int_seqs, enforce_sorted=enforce_sorted)
    tok_embed = self.tok_embed(packed_seq.data)
    packed_input = PackedSequence(data=tok_embed, batch_sizes=packed_seq.batch_sizes, 
                    sorted_indices=packed_seq.sorted_indices, unsorted_indices=packed_seq.unsorted_indices)

    packed_out, state = self.rnn(packed_input, state)
    unpacked_out, _ = pad_packed_sequence(packed_out)
    out_logits = self.out_pred(unpacked_out)[:-1, :, :].view(-1, len(self.vocab))
    target_seq = pad_packed_sequence(packed_seq)[0][1:, :].view(-1)

    loss = F.cross_entropy(out_logits, target_seq, ignore_index=self.tok_pad, reduction='none').view(-1, len(expr_list))
    ll = -torch.sum(loss, 0).view(-1, 1)        
    return ll

  def setup_init_tokens(self, init_states, num_samples, device):
    cur_tok = [self.tok_start for _ in range(num_samples)]        
    tokens = [[self.inv_map[t]] for t in cur_tok]
    cur_tok = torch.LongTensor(cur_tok).to(device).view(-1, 1)
    return cur_tok, tokens

  def beam_search(self, state, beam_size, max_len, cur_sizes=None, init_states=None, init_ll=None):
    state = self.get_init_state(state)
    num_samples = state[0].shape[1]
    device = state[0].device
    if cur_sizes is None:
      cur_sizes = [1] * num_samples
      batch_size = num_samples
    else:
      batch_size = len(cur_sizes)
    ll = torch.zeros(num_samples, 1).to(device) if init_ll is None else init_ll
    
    fsm_state = torch.LongTensor([0] * num_samples).to(device)

    cur_tok, _ = self.setup_init_tokens(init_states, num_samples, device)
    ones = torch.LongTensor([1] * beam_size * batch_size).to(device)
    all_toks = cur_tok
    ancestors = torch.LongTensor(list(range(num_samples))).to(device)
    n_steps = 0
    while True:
      n_steps += 1
      if n_steps > 100:
        break
      cur_mask = self.decision_mask[fsm_state]
      cur_embed = self.tok_embed(cur_tok.view(1, -1))
      out, state = self.rnn(cur_embed, state)
      out_logits = self.out_pred(out.squeeze(0))
      out_logits = out_logits * cur_mask + (1 - cur_mask) * N_INF
      out_logprob = F.log_softmax(out_logits, dim=-1)

      # do one step (topk)
      raw_scores = out_logprob + ll
      predecessors, pred_toks, ll, cur_sizes = beam_step(raw_scores, cur_sizes, beam_size)      
      ancestors = ancestors[predecessors]
      fsm_state = self.state_trans[fsm_state[predecessors].view(-1), pred_toks.view(-1)]
      cur_tok = pred_toks
      all_toks = torch.cat([all_toks[predecessors], pred_toks.view(-1, 1)], dim=-1)
      state = (state[0][:, predecessors, :], state[1][:, predecessors, :])
      if torch.all(fsm_state == 1).item():
        break
    tokens = []
    all_toks = all_toks.data.cpu().numpy()
    for i in range(all_toks.shape[0]):
      cur_prog = [self.inv_map[j] for j in all_toks[i] if j != self.tok_stop and j != self.tok_start]
      tokens.append(cur_prog)
    return ll, tokens, cur_sizes, ancestors

  def forward(self, state, expr_list, cooked_data=None):
    state = self.get_init_state(state)
    ll = self.get_likelihood(state, expr_list, enforce_sorted=cooked_data is not None, cooked_data=cooked_data)
    return ll
