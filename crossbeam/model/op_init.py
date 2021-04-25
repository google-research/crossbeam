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


class PoolingState(nn.Module):
  def __init__(self, state_dim, pool_method):
    super(PoolingState, self).__init__()
    self.state_dim = state_dim
    self.pool_method = pool_method
    self.proj = nn.Linear(state_dim * 3, state_dim)

  def forward(self, io_embed, value_embed, value_mask=None):
    torch_pool = getattr(torch, self.pool_method, None)
    if self.pool_method == 'max' or self.pool_method == 'min':
      pool_method = lambda x: torch_pool(x, dim=0, keepdims=True)[0]
    else:
      pool_method = lambda x: torch_pool(x, dim=0, keepdims=True)
  
    io_state = pool_method(io_embed)

    if value_mask is None:
      value_state = pool_method(value_embed)
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


class OpPoolingState(nn.Module):
  def __init__(self, ops, state_dim, pool_method):
    super(OpPoolingState, self).__init__()
    self.ops = ops
    self.state_dim = state_dim
    self.op_specific_mod = nn.ModuleList([PoolingState(self.state_dim, pool_method) for _ in range(len(self.ops))])
    self.op_idx_map = {repr(op): i for i, op in enumerate(self.ops)}

  def forward(self, io_embed, value_embed, op, value_mask=None):
    mod = self.op_specific_mod[self.op_idx_map[repr(op)]]
    return mod(io_embed, value_embed, value_mask)
