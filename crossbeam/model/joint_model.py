import torch
import torch.nn as nn
import torch.nn.functional as F

from crossbeam.model.op_arg import LSTMArgSelector
from crossbeam.model.op_init import OpPoolingState
from crossbeam.model.encoder import CharIOLSTMEncoder, CharValueLSTMEncoder


class JointModel(nn.Module):
  def __init__(self, args, input_table, output_table, value_table, operations):
    super(JointModel, self).__init__()
    self.device = 'cpu'
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=args.embed_dim)
    self.val = CharValueLSTMEncoder(value_table, hidden_size=args.embed_dim)
    self.arg = LSTMArgSelector(hidden_size=args.embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=args.step_score_func,
                               step_score_normalize=args.score_normed)
    self.init = OpPoolingState(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')

  def set_device(self, device):
    self.device = device
    self.io.set_device(device)
    self.val.set_device(device)
    self.arg.set_device(device)
