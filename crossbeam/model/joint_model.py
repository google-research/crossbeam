import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from crossbeam.model.op_arg import LSTMArgSelector, OpSpecificLSTMSelector
from crossbeam.model.op_init import PoolingState, OpPoolingState, OpExplicitPooling
from crossbeam.model.encoder import CharIOLSTMEncoder, CharValueLSTMEncoder, IntIOEncoder, IntValueEncoder, ValueAndOpEncoder


class JointModel(nn.Module):
  def __init__(self, args, input_table, output_table, value_table, operations):
    super(JointModel, self).__init__()
    self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=args.embed_dim)
    val = CharValueLSTMEncoder(value_table, hidden_size=args.embed_dim)
    self.op_in_beam = args.op_in_beam
    if args.op_in_beam:
      self.val = ValueAndOpEncoder(operations, val)
      arg_mod = LSTMArgSelector
      self.init = PoolingState(state_dim=args.embed_dim, pool_method='mean')
    else:
      self.val = val
      if 'op' in args.model_type:
        arg_mod = partial(OpSpecificLSTMSelector, operations)
        init_mod = OpExplicitPooling
      else:
        arg_mod = LSTMArgSelector
        init_mod = OpPoolingState
      self.init = init_mod(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')
    self.arg = arg_mod(hidden_size=args.embed_dim,
                       mlp_sizes=[256, 1],
                       step_score_func=args.step_score_func,
                       step_score_normalize=args.score_normed)


class IntJointModel(nn.Module):
  def __init__(self, args, input_range, output_range, value_range, operations):
    super(IntJointModel, self).__init__()
    self.io = IntIOEncoder(input_range, output_range, args.max_num_inputs, hidden_size=args.embed_dim)
    self.val = IntValueEncoder(value_range, hidden_size=args.embed_dim, num_samples=args.max_num_examples)
    self.arg = LSTMArgSelector(hidden_size=args.embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=args.step_score_func,
                               step_score_normalize=args.score_normed)
    self.init = OpPoolingState(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')
