from typing import Callable
import numpy as np
import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence

from crossbeam.model.base import CharSeqEncoder, pad_sequence
from crossbeam.model.util import CharacterTable


def pad_int_seq(int_seqs, device):
  int_seqs = [torch.LongTensor(x) for x in int_seqs]
  lengths = [v.size(0) for v in int_seqs]
  padded = pad_sequence(int_seqs)
  return padded.to(device), lengths


def get_int_mapped(v, int_range):
  if v >= int_range[0] and v < int_range[1]:
    return v - int_range[0]
  d = int_range[1] - int_range[0]
  if v < int_range[0]:  # underflow
    return d
  else:  # overflow
    return d + 1


class IntIOEncoder(nn.Module):
  def __init__(self, input_range, output_range, num_input_vars, hidden_size):
    super(IntIOEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_range = input_range
    self.output_range = output_range
    self.num_input_vars = num_input_vars
    self.input_int_embed = nn.Embedding(input_range[1] - input_range[0] + 2, hidden_size)
    self.output_int_embed = nn.Embedding(output_range[1] - output_range[0] + 2, hidden_size)
    self.input_linear = nn.Linear(self.num_input_vars * hidden_size, hidden_size)

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    sample_scatter_idx = []
    list_input = []
    list_output = []
    for sample_idx, (inputs_dict, outputs) in enumerate(zip(list_inputs_dict, list_outputs)):
      n_ios = len(outputs)
      cur_input = torch.LongTensor(n_ios, self.num_input_vars)
      v_idx = 0
      for _, input_value in inputs_dict.items():        
        for i in range(n_ios):
          cur_input[i, v_idx] = get_int_mapped(input_value[i], self.input_range)
        v_idx += 1
      assert v_idx == self.num_input_vars
      # n_ios * num_inputs * hidden_size
      list_input.append(cur_input)
      cur_output = torch.LongTensor(n_ios)
      for i in range(n_ios):
        cur_output[i] = get_int_mapped(outputs[i], self.output_range)
      list_output.append(cur_output)
      sample_scatter_idx += [sample_idx] * n_ios

    input_embed = self.input_int_embed(torch.cat(list_input).to(device)).view(-1, self.num_input_vars * self.hidden_size)
    output_embed = self.output_int_embed(torch.cat(list_output).to(device))
    input_embed = self.input_linear(input_embed)
    cat_embed = torch.cat((input_embed, output_embed), dim=-1)
    if needs_scatter_idx:
      sample_scatter_idx = torch.LongTensor(sample_scatter_idx).to(device)
      return cat_embed, sample_scatter_idx
    else:
      return cat_embed


class IntValueEncoder(nn.Module):
  def __init__(self, value_range, hidden_size, num_samples):
    super(IntValueEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.value_range = value_range
    self.num_samples = num_samples
    self.value_int_embed = nn.Embedding(value_range[1] - value_range[0] + 2, hidden_size)
    self.value_linear = nn.Linear(self.num_samples * hidden_size, hidden_size)

  def forward(self, all_values, device):
    int_vals = torch.LongTensor(len(all_values), self.num_samples)
    for i, v in enumerate(all_values):
      assert len(v.values) == self.num_samples
      for j in range(len(v.values)):
        int_vals[i, j] = get_int_mapped(v[j], self.value_range)
    concat_embed = self.value_int_embed(int_vals.to(device)).view(-1, self.num_samples * self.hidden_size)
    val_embed = self.value_linear(concat_embed)
    return val_embed


class CharIOLSTMEncoder(nn.Module):
  def __init__(self, input_char_table, output_char_table, hidden_size,
               to_string: Callable = repr):
    super(CharIOLSTMEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.to_string = to_string
    self.input_char_table = input_char_table
    self.output_char_table = output_char_table
    self.input_encoder = CharSeqEncoder(self.input_char_table.vocab_size, self.hidden_size)
    self.output_encoder = CharSeqEncoder(self.output_char_table.vocab_size, self.hidden_size)

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    sample_scatter_idx = []
    list_input = []
    list_output = []
    for sample_idx, (inputs_dict, outputs) in enumerate(zip(list_inputs_dict, list_outputs)):
      cur_input = [''] * len(outputs)
      for _, input_value in inputs_dict.items():
        for i in range(len(cur_input)):
          cur_input[i] += self.to_string(input_value[i]) + ','
      list_input += cur_input
      cur_output = [self.to_string(x) for x in outputs]
      list_output += cur_output
      sample_scatter_idx += [sample_idx] * len(outputs)

    list_int_i = []
    list_int_o = []
    for l, tab, lst in [(list_input, self.input_char_table, list_int_i),
                        (list_output, self.output_char_table, list_int_o)]:
      for obj in l:
        tokens = tab.encode(obj)
        lst.append(tokens)

    padded_i, len_i = pad_int_seq(list_int_i, device)
    padded_o, len_o = pad_int_seq(list_int_o, device)

    input_embed = self.input_encoder(padded_i, len_i)
    output_embed = self.output_encoder(padded_o, len_o)
    cat_embed = torch.cat((input_embed, output_embed), dim=-1)
    if needs_scatter_idx:
      sample_scatter_idx = torch.LongTensor(sample_scatter_idx).to(device)
      return cat_embed, sample_scatter_idx
    else:
      return cat_embed


class CharValueLSTMEncoder(nn.Module):
  def __init__(self, val_char_table, hidden_size, to_string: Callable = repr):
    super(CharValueLSTMEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.to_string = to_string
    self.val_char_table = val_char_table
    self.val_encoder = CharSeqEncoder(self.val_char_table.vocab_size, self.hidden_size)
  
  def forward(self, all_values, device):
    list_values = [self.to_string(x) for x in all_values]
    list_int_vals = [self.val_char_table.encode(x) for x in list_values]
    padded_i, len_i = pad_int_seq(list_int_vals, device)
    val_embed = self.val_encoder(padded_i, len_i)
    return val_embed
