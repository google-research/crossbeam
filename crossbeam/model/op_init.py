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

from random import sample
from crossbeam.dsl import value
from crossbeam.dsl.value import Value
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from crossbeam.model.base import MLP
from torch_scatter import scatter_mean, scatter_max


class IOPoolProjSummary(nn.Module):
  def __init__(self, embed_dim, pool_method):
    super(IOPoolProjSummary, self).__init__()
    self.embed_merge = nn.Linear(2 * embed_dim, embed_dim)    
    if pool_method == 'max':
      self.agg_func = lambda x, idx: scatter_max(x, idx, dim=0)[0]
    elif pool_method == 'mean':
      self.agg_func = lambda x, idx: scatter_mean(x, idx, dim=0)
    else:
      raise NotImplementedError

  def forward(self, io_concat_embed, scatter_idx=None):
    if scatter_idx is not None:
      pooled_io = self.agg_func(io_concat_embed, scatter_idx)
    else:
      pooled_io = io_concat_embed
    return self.embed_merge(pooled_io)


def batch_pooling(pool_method, io_embed, io_scatter, value_embed, value_indices, sample_indices=None, io_gather=None):
  if pool_method == 'max':
    agg_func = lambda x, idx: scatter_max(x, idx, dim=0)[0]
  elif pool_method == 'mean':
    agg_func = lambda x, idx: scatter_mean(x, idx, dim=0)
  else:
    raise ValueError('unknown pooling %s' % pool_method)
  io_state = agg_func(io_embed, io_scatter)
  if sample_indices is not None:
    io_state = io_state[sample_indices]
    value_indices = [value_indices[v] for v in sample_indices]

  joint_val_embed = value_embed[torch.cat(value_indices, dim=0)]
  val_scatter = []
  for i, v in enumerate(value_indices):
    val_scatter += [i] * v.shape[0]
  val_scatter = torch.LongTensor(val_scatter).to(io_state.device)
  value_state = agg_func(joint_val_embed, val_scatter)
  if io_gather is not None:
    io_state = io_state[io_gather]
  joint_state = torch.cat((io_state, value_state), dim=1)
  return joint_state


class PoolingState(nn.Module):
  def __init__(self, state_dim, pool_method):
    super(PoolingState, self).__init__()
    self.state_dim = state_dim
    self.pool_method = pool_method
    self.proj = nn.Linear(state_dim * 3, state_dim)

  @property
  def fn_pool(self):
    torch_pool = getattr(torch, self.pool_method, None)
    if self.pool_method == 'max' or self.pool_method == 'min':
      pool_method = lambda x: torch_pool(x, dim=0, keepdims=True)[0]
    else:
      pool_method = lambda x: torch_pool(x, dim=0, keepdims=True)
    return pool_method

  def forward(self, io_embed, value_embed, dummy_op=None, value_mask=None):
    io_state = self.fn_pool(io_embed)

    if value_mask is None:
      value_state = self.fn_pool(value_embed)
    else:
      value_mask = torch.unsqueeze(value_mask, dim=1)
      if self.pool_method == 'mean':
        value_state = torch.sum(value_embed * value_mask, dim=0, keepdims=True)
        value_state = value_state / (torch.sum(value_mask) + 1e-10)
      elif self.pool_method == 'max':
        value_state = value_embed * value_mask + -1e10 * (1 - value_mask)
        value_state, _ = torch.max(value_state, dim=0, keepdims=True)
      else:
        raise NotImplementedError
    joint_state = torch.cat((io_state, value_state), dim=1)
    return self.proj(joint_state)

  def batch_forward(self, io_embed, io_scatter, value_embed, value_indices, sample_indices=None, io_gather=None):
    joint_state = batch_pooling(self.pool_method, io_embed, io_scatter, value_embed, value_indices, sample_indices, io_gather)
    return self.proj(joint_state)


class OpPoolingState(nn.Module):
  def __init__(self, ops, state_dim, pool_method):
    super(OpPoolingState, self).__init__()
    self.ops = ops
    self.state_dim = state_dim
    self.pool_method = pool_method
    self.op_specific_mod = nn.ModuleList([PoolingState(self.state_dim, pool_method) for _ in range(len(self.ops))])
    self.op_idx_map = {repr(op): i for i, op in enumerate(self.ops)}

  def forward(self, io_embed, value_embed, op, value_mask=None):
    mod = self.op_specific_mod[self.op_idx_map[repr(op)]]
    return mod(io_embed, value_embed, value_mask)

  def batch_forward(self, io_embed, io_scatter, value_embed, value_indices, op, sample_indices=None, io_gather=None):
    if isinstance(op, list):
      joint_state = batch_pooling(self.pool_method, io_embed, io_scatter, value_embed, value_indices, sample_indices, io_gather)
      assert len(op) == joint_state.shape[0]
      weights = [self.op_specific_mod[self.op_idx_map[repr(o)]].proj.weight for o in op]
      bias = [self.op_specific_mod[self.op_idx_map[repr(o)]].proj.bias for o in op]
      joint_state = joint_state.unsqueeze(-1)
      weights = torch.stack(weights, dim=0)
      bias = torch.stack(bias, dim=0)
      h = torch.bmm(weights, joint_state).squeeze(-1)
      return h + bias
    mod = self.op_specific_mod[self.op_idx_map[repr(op)]]
    return mod.batch_forward(io_embed, io_scatter, value_embed, value_indices, sample_indices, io_gather)
