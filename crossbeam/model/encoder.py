from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence

from crossbeam.model.base import CharSeqEncoder, pad_sequence
from crossbeam.model.util import CharacterTable


def pad_int_seq(int_seqs):
  int_seqs = [torch.LongTensor(x) for x in int_seqs]
  lengths = [v.size(0) for v in int_seqs]
  return pad_sequence(int_seqs), lengths


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

  def forward(self, inputs_dict, outputs):
    list_input = [''] * len(outputs)
    for _, input_value in inputs_dict.items():
      for i in range(len(list_input)):
        list_input[i] += self.to_string(input_value[i]) + ','
    list_output = [self.to_string(x) for x in outputs]
    list_int_i = []
    list_int_o = []
    for l, tab, lst in [(list_input, self.input_char_table, list_int_i),
                        (list_output, self.output_char_table, list_int_o)]:
      for obj in l:
        tokens = tab.encode(obj)
        lst.append(tokens)

    padded_i, len_i = pad_int_seq(list_int_i)
    padded_o, len_o = pad_int_seq(list_int_o)

    input_embed = self.input_encoder(padded_i, len_i)
    output_embed = self.output_encoder(padded_o, len_o)
    cat_embed = torch.cat((input_embed, output_embed), dim=-1)
    return cat_embed


class CharValueLSTMEncoder(nn.Module):
  def __init__(self, val_char_table, hidden_size, to_string: Callable = repr):
    super(CharValueLSTMEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.to_string = to_string
    self.val_char_table = val_char_table
    self.val_encoder = CharSeqEncoder(self.val_char_table.vocab_size, self.hidden_size)
  
  def forward(self, all_values):
    list_values = [self.to_string(x) for x in all_values]
    list_int_vals = [self.val_char_table.encode(x) for x in list_values]
    padded_i, len_i = pad_int_seq(list_int_vals)
    val_embed = self.val_encoder(padded_i, len_i)
    return val_embed