# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from crossbeam.dsl.operation_base import OperationBase
from crossbeam.dsl import value as value_module
from crossbeam.model.base import CharSeqEncoder, pad_sequence, _param_init
from crossbeam.model.util import CharacterTable
from crossbeam.algorithm import property_signatures
from crossbeam.property_signatures import property_signatures as deepcoder_propsig
from crossbeam.algorithm.variables import MAX_NUM_FREE_VARS

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


def get_pad(int_range):
  d = int_range[1] - int_range[0]
  return d + 2


class IntIOEncoder(nn.Module):
  def __init__(self, input_range, output_range, num_input_vars, hidden_size):
    super(IntIOEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_range = input_range
    self.output_range = output_range
    self.num_input_vars = num_input_vars
    self.input_int_embed = nn.Embedding(input_range[1] - input_range[0] + 3, hidden_size)
    self.output_int_embed = nn.Embedding(output_range[1] - output_range[0] + 3, hidden_size)
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
      while v_idx < self.num_input_vars:
        for i in range(n_ios):
          cur_input[i, v_idx] = get_pad(self.input_range)
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
    self.value_int_embed = nn.Embedding(value_range[1] - value_range[0] + 3, hidden_size)
    self.value_linear = nn.Linear(self.num_samples * hidden_size, hidden_size)

  def forward(self, all_values, device, output_values):
    del output_values  # unused
    int_vals = torch.LongTensor(len(all_values), self.num_samples)
    for i, v in enumerate(all_values):
      assert len(v.values) <= self.num_samples
      for j in range(len(v.values)):
        int_vals[i, j] = get_int_mapped(v[j], self.value_range)
      for j in range(len(v.values), self.num_samples):
        int_vals[i, j] = get_pad(self.value_range)
    concat_embed = self.value_int_embed(int_vals.to(device)).view(-1, self.num_samples * self.hidden_size)
    val_embed = self.value_linear(concat_embed)
    return val_embed


class BustlePropSigIOEncoder(nn.Module):
  def __init__(self, max_num_inputs, hidden_size):
    super(BustlePropSigIOEncoder, self).__init__()
    self.max_num_inputs = max_num_inputs
    self.num_sigs = property_signatures.NUM_SINGLE_VALUE_PROPERTIES + property_signatures.NUM_COMPARISON_PROPERTIES
    self.feat_dim = self.num_sigs * self.max_num_inputs + property_signatures.NUM_SINGLE_VALUE_PROPERTIES
    self.summary_embed = nn.Embedding(5, 2)
    self.mlp = nn.Sequential(
      nn.Linear(self.feat_dim * 2, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size * 2)
    )

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    num_tasks = len(list_outputs)
    feature_list = []
    for inputs_dict, outputs in zip(list_inputs_dict, list_outputs):
      cur_input = []
      for input_name, input_value in inputs_dict.items():
        cur_input.append(value_module.InputVariable(input_value, name=input_name))
      signature = property_signatures.compute_example_signature(cur_input, value_module.OutputValue(outputs))
      cur_feats = [int(sig) for sig in signature]
      feature_list.append(cur_feats)
    feat_mat = torch.LongTensor(feature_list).to(device)
    feat_embed = self.summary_embed(feat_mat).view(-1, self.feat_dim * 2)
    io_embed = self.mlp(feat_embed)
    if needs_scatter_idx:
      sample_scatter_idx = torch.arange(num_tasks).to(device)
      return io_embed, sample_scatter_idx
    else:
      return io_embed


class Signature(nn.Module):
  def __init__(self, len_signature):
    super(Signature, self).__init__()
    self.tuple_length = deepcoder_propsig.SIGNATURE_TUPLE_LENGTH
    self.embed_length = 2
    self.frac_applicable_embed = nn.Embedding(12, self.embed_length)
    self.frac_tf_embed = nn.Embedding(12, self.embed_length)
    self.len_signature = len_signature

  def quantize(self, sig):
    return torch.where(sig < 1e-8, 0, (sig * 10).floor().long() + 1)

  def forward(self, list_signatures, device):
    # shape: [num signatures, self.len_signature, self.tuple_length]
    signatures = torch.FloatTensor(list_signatures).to(device)
    signatures = self.quantize(signatures).long()
    # shape: [num signatures, self.len_signature, self.tuple_length, self.embed_length]
    embed = torch.cat([
        self.frac_applicable_embed(signatures[:, :, 0:1]),
        self.frac_tf_embed(signatures[:, :, 1:2])
    ], dim=2)
    # shape: [num signatures, self.len_signature * self.tuple_length * self.embed_length]
    flat_embed = embed.view(embed.shape[0], -1)
    return flat_embed


class LambdaSigIOEncoder(Signature):
  def __init__(self, max_num_inputs, hidden_size):
    super(LambdaSigIOEncoder, self).__init__(deepcoder_propsig.IO_EXAMPLES_SIGNATURE_LENGTH)
    self.max_num_inputs = max_num_inputs
    self.mlp = nn.Sequential(
      nn.Linear(self.len_signature * self.tuple_length * self.embed_length, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size * 2)
    )

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    num_tasks = len(list_outputs)
    list_signatures = []
    for inputs_dict, outputs in zip(list_inputs_dict, list_outputs):
      cur_input = []
      for input_name, input_value in inputs_dict.items():
        cur_input.append(value_module.InputVariable(input_value, name=input_name))
      signature = deepcoder_propsig.property_signature_io_examples(cur_input, value_module.OutputValue(outputs), fixed_length=True)
      list_signatures.append(signature)

    feat_embed = super(LambdaSigIOEncoder, self).forward(list_signatures, device)
    io_embed = self.mlp(feat_embed)
    if needs_scatter_idx:
      sample_scatter_idx = torch.arange(num_tasks).to(device)
      return io_embed, sample_scatter_idx
    else:
      return io_embed


class PropSigIOEncoder(nn.Module):
  def __init__(self, max_num_inputs, hidden_size):
    super(PropSigIOEncoder, self).__init__()
    self.max_num_inputs = max_num_inputs
    self.num_sigs = property_signatures.NUM_SINGLE_VALUE_PROPERTIES + property_signatures.NUM_COMPARISON_PROPERTIES
    self.feat_dim = self.num_sigs * self.max_num_inputs + property_signatures.NUM_SINGLE_VALUE_PROPERTIES
    self.mlp = nn.Sequential(
      nn.Linear(self.feat_dim * 5, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size * 2)
    )

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    num_tasks = len(list_outputs)
    feat_mat = torch.zeros(num_tasks, 5 * self.feat_dim)
    for sample_idx, (inputs_dict, outputs) in enumerate(zip(list_inputs_dict, list_outputs)):
      cur_input = []
      for input_name, input_value in inputs_dict.items():
        cur_input.append(value_module.InputVariable(input_value, name=input_name))
      signature = property_signatures.compute_example_signature(cur_input, value_module.OutputValue(outputs))
      for i, sig in enumerate(signature):
        feat_mat[sample_idx, i * 5 + int(sig)] = 1.0
    feat_mat = feat_mat.to(device)
    io_embed = self.mlp(feat_mat)
    if needs_scatter_idx:
      sample_scatter_idx = torch.arange(num_tasks).to(device)
      return io_embed, sample_scatter_idx
    else:
      return io_embed


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


class CharAndPropSigIOEncoder(nn.Module):
  def __init__(self, max_num_inputs, input_char_table, output_char_table, hidden_size,
               to_string: Callable = repr):
    super(CharAndPropSigIOEncoder, self).__init__()
    self.char_io_encoder = CharIOLSTMEncoder(input_char_table, output_char_table, hidden_size, to_string)
    self.prop_sig_encoder = PropSigIOEncoder(max_num_inputs, hidden_size)
    self.merge_embed = nn.Linear(4 * hidden_size, 2 * hidden_size)

  def forward(self, list_inputs_dict, list_outputs, device, needs_scatter_idx=False):
    char_cat_embed, sample_scatter_idx = self.char_io_encoder(list_inputs_dict, list_outputs, device, needs_scatter_idx=True)
    sig_cat_embed = self.prop_sig_encoder(list_inputs_dict, list_outputs, device)
    repeat_sig_cat = sig_cat_embed[sample_scatter_idx]
    merged_embed = torch.cat((char_cat_embed, repeat_sig_cat), dim=-1)
    merged_embed = self.merge_embed(merged_embed)
    if needs_scatter_idx:
      return merged_embed, sample_scatter_idx
    else:
      return merged_embed


class CharValueLSTMEncoder(nn.Module):
  def __init__(self, val_char_table, hidden_size, to_string: Callable = repr):
    super(CharValueLSTMEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.to_string = to_string
    self.val_char_table = val_char_table
    self.val_encoder = CharSeqEncoder(self.val_char_table.vocab_size, self.hidden_size)
  
  def forward(self, all_values, device, output_values=None):
    list_values = [self.to_string(x) for x in all_values]
    list_int_vals = [self.val_char_table.encode(x) for x in list_values]
    padded_i, len_i = pad_int_seq(list_int_vals, device)
    val_embed = self.val_encoder(padded_i, len_i)
    return val_embed


class PropSigValueEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(PropSigValueEncoder, self).__init__()
    self.num_sigs = property_signatures.NUM_SINGLE_VALUE_PROPERTIES + property_signatures.NUM_COMPARISON_PROPERTIES
    self.mlp = nn.Sequential(
      nn.Linear(self.num_sigs * 5, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size)
    )

  def forward(self, all_values, device, output_values):
    val_feat = torch.zeros(len(all_values), self.num_sigs * 5)
    for v_idx, v in enumerate(all_values):
      signatures = property_signatures.compute_value_signature(v, output_values)
      for i, sig in enumerate(signatures):
        val_feat[v_idx, i * 5 + int(sig)] = 1.0
    val_feat = val_feat.to(device)
    val_embed = self.mlp(val_feat)
    return val_embed


class BustlePropSigValueEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(BustlePropSigValueEncoder, self).__init__()
    self.num_sigs = property_signatures.NUM_SINGLE_VALUE_PROPERTIES + property_signatures.NUM_COMPARISON_PROPERTIES
    self.embed_dim = 2
    self.summary_embed = nn.Embedding(5, self.embed_dim)
    self.mlp = nn.Sequential(
      nn.Linear(self.num_sigs * self.embed_dim, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size)
    )

  def forward(self, all_values, device, output_values):
    feat_list = []
    for v in all_values:
      signatures = property_signatures.compute_value_signature(v, output_values)
      cur_feat = [int(sig) for sig in signatures]
      feat_list.append(cur_feat)
    feat_list = torch.LongTensor(feat_list).to(device)
    feat_embed = self.summary_embed(feat_list).view(-1, self.num_sigs * self.embed_dim)
    val_embed = self.mlp(feat_embed)
    return val_embed


class IndexedConcat(torch.autograd.Function):

  @staticmethod
  def forward(ctx, *tensors):
    num_tensors = len(tensors)
    list_embedding = tensors[:num_tensors // 2]
    list_idx = tensors[num_tensors // 2:]
    num_rows = sum(x.shape[0] for x in list_idx)
    out_tensor = list_embedding[0].new(num_rows, list_embedding[0].shape[1]).contiguous()
    for i in range(len(list_idx)):
      out_tensor[list_idx[i]] = list_embedding[i]
    ctx.save_for_backward(*list_idx)
    return out_tensor

  @staticmethod
  def backward(ctx, grad_out):
    list_idx = ctx.saved_tensors
    list_grads = []
    for i in range(len(list_idx)):
      cur_grad = grad_out[list_idx[i]]
      list_grads.append(cur_grad)
    for i in range(len(list_idx)):
      list_grads.append(None)
    return tuple(list_grads)

indexed_concat = IndexedConcat.apply


class LambdaSigValueEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(LambdaSigValueEncoder, self).__init__()
    self.concrete_signature = Signature(deepcoder_propsig.CONCRETE_SIGNATURE_LENGTH)
    self.lambda_signature = Signature(deepcoder_propsig.LAMBDA_SIGNATURE_LENGTH)
    tuple_length = self.concrete_signature.tuple_length
    embed_length = self.concrete_signature.embed_length
    self.freevar_embed = nn.Parameter(torch.zeros(MAX_NUM_FREE_VARS, hidden_size))
    nn.init.xavier_uniform_(self.freevar_embed)
    self.mlp_concrete = nn.Sequential(
      nn.Linear(self.concrete_signature.len_signature * tuple_length * embed_length, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size)
    )
    self.mlp_lambda = nn.Sequential(
      nn.Linear(self.lambda_signature.len_signature * tuple_length * embed_length, hidden_size * 2),
      nn.ReLU(),
      nn.Linear(hidden_size * 2, hidden_size)
    )

  def forward_with_signatures(self, all_values, device, list_normal_signatures):
    list_special_vars = []
    feat_special_vars = []
    for vidx, v in enumerate(all_values):
      if isinstance(v, value_module.FreeVariable):
        list_special_vars.append(vidx)
        feat_special_vars.append(self.freevar_embed[int(v.name[1:]) - 1].view(1, -1))
    if len(list_normal_signatures) == 0:
      assert len(list_special_vars)
      return torch.cat(feat_special_vars, dim=0)

    list_of_list_signatures = [[], []]
    list_of_list_sidx = [[], []]
    for sidx, (nv, signature) in enumerate(list_normal_signatures):
      selector = int(nv > 0)
      list_of_list_signatures[selector].append(signature)
      list_of_list_sidx[selector].append(sidx)

    if len(list_of_list_signatures[0]):
      concrete_sig_embed = self.concrete_signature(list_of_list_signatures[0], device)
      concrete_sig_embed = self.mlp_concrete(concrete_sig_embed)
      if len(list_of_list_signatures[1]) == 0:
        normal_sig_embed = concrete_sig_embed
    if len(list_of_list_signatures[1]):
      lambda_sig_embed = self.lambda_signature(list_of_list_signatures[1], device)
      lambda_sig_embed = self.mlp_lambda(lambda_sig_embed)
      if len(list_of_list_signatures[0]) == 0:
        normal_sig_embed = lambda_sig_embed
    if len(list_of_list_signatures[0]) and len(list_of_list_signatures[1]):
      normal_sig_embed = indexed_concat(concrete_sig_embed, lambda_sig_embed,
                                        torch.LongTensor(list_of_list_sidx[0]).to(device),
                                        torch.LongTensor(list_of_list_sidx[1]).to(device))

    if len(list_special_vars) == 0:
      all_embed = normal_sig_embed
    else:
      special_embed = torch.cat(feat_special_vars, dim=0)
      # assume the free vars appear in a contiguous fashion, so that concatenation is easier
      first_free_var = min(list_special_vars)
      assert len(list_special_vars) == max(list_special_vars) - first_free_var + 1
      part1, part2 = torch.split(normal_sig_embed, [first_free_var, normal_sig_embed.shape[0] - first_free_var])
      all_embed = torch.cat([part1, special_embed, part2], dim=0)
    assert all_embed.shape[0] == len(all_values)
    return all_embed

  def forward(self, all_values, device, output_values, need_signatures=False):
    list_normal_signatures = []
    for v in all_values:
      if not isinstance(v, value_module.FreeVariable):
        signature = deepcoder_propsig.property_signature_value(v, output_values, fixed_length=True)
        list_normal_signatures.append((v.num_free_variables, signature))
    all_embed = self.forward_with_signatures(all_values, device, list_normal_signatures)
    if need_signatures:
      return all_embed, list_normal_signatures
    else:
      return all_embed


class CharAndPropSigValueEncoder(nn.Module):
  def __init__(self, val_char_table, hidden_size, to_string: Callable = repr):
    super(CharAndPropSigValueEncoder, self).__init__()
    self.char_encoder = CharValueLSTMEncoder(val_char_table, hidden_size, to_string)
    self.sig_encoder = PropSigValueEncoder(hidden_size)
    self.merge_embed = nn.Linear(2 * hidden_size, hidden_size)

  def forward(self, all_values, device, output_values):
    char_embed = self.char_encoder(all_values, device)
    sig_embed = self.sig_encoder(all_values, device, output_values)
    embed = torch.cat((char_embed, sig_embed), dim=-1)
    embed = self.merge_embed(embed)
    return embed


class ValueAndOpEncoder(nn.Module):
  def __init__(self, ops, val_encoder):
    super(ValueAndOpEncoder, self).__init__()
    self.val_encoder = val_encoder
    self.num_ops = len(ops)
    self.op_embedding = nn.Parameter(torch.zeros(self.num_ops, val_encoder.hidden_size))
    _param_init(self.op_embedding)    

  def forward(self, all_values, device):
    if isinstance(all_values[0], OperationBase):
      assert all([isinstance(x, OperationBase) for x in all_values[:self.num_ops]])
      val_embed = self.val_encoder(all_values[self.num_ops:], device)
      val_embed = torch.cat((self.op_embedding, val_embed), dim=0)
    else:
      val_embed = self.val_encoder(all_values, device)    
    return val_embed


class DummyWeightEncoder(nn.Module):
  def __init__(self):
    super(DummyWeightEncoder, self).__init__()

  def forward(self, value_embed, all_values):
    return value_embed


class ValueWeightEncoder(nn.Module):
  def __init__(self, hidden_size, max_weight=20):
    super(ValueWeightEncoder, self).__init__()
    self.max_weight = max_weight
    self.weight_embedding = nn.Embedding(max_weight + 1, hidden_size)

  def forward(self, value_embed, all_weights):
    wids = [min(v, self.max_weight + 1) - 1 for v in all_weights]
    wids = torch.LongTensor(wids).to(value_embed.device)
    w_embed = self.weight_embedding(wids)
    value_embed = value_embed + w_embed
    return value_embed
