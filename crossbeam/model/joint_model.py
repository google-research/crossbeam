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

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from crossbeam.model.op_arg import LSTMArgSelector
from crossbeam.model.op_init import PoolingState, OpPoolingState
from crossbeam.model.encoder import CharIOLSTMEncoder, CharValueLSTMEncoder, PropSigIOEncoder, PropSigValueEncoder, IntIOEncoder, IntValueEncoder, ValueAndOpEncoder
from crossbeam.model.encoder import CharAndPropSigIOEncoder, CharAndPropSigValueEncoder, BustlePropSigIOEncoder, BustlePropSigValueEncoder, DummyWeightEncoder, ValueWeightEncoder


class JointModel(nn.Module):
  def __init__(self, args, input_table, output_table, value_table, operations):
    super(JointModel, self).__init__()
    if args.io_encoder == 'char':
      self.io = CharIOLSTMEncoder(input_table, output_table, hidden_size=args.embed_dim)
      print('io encoder: char')
    elif args.io_encoder == 'signature':
      self.io = PropSigIOEncoder(args.max_num_inputs, hidden_size=args.embed_dim)
      print('io encoder: signature')
    elif args.io_encoder == 'char_sig':
      self.io = CharAndPropSigIOEncoder(args.max_num_inputs, input_table, output_table, hidden_size=args.embed_dim)
      print('io encoder: char+sig')
    elif args.io_encoder == 'bustle_sig':
      self.io = BustlePropSigIOEncoder(args.max_num_inputs, hidden_size=args.embed_dim)
      print('io encoder: bustle signature')
    else:
      raise ValueError('unknown io encoder %s' % args.io_encoder)
    if args.value_encoder == 'char':
      val = CharValueLSTMEncoder(value_table, hidden_size=args.embed_dim)
      print('value encoder: char')
    elif args.value_encoder == 'signature':
      val = PropSigValueEncoder(hidden_size=args.embed_dim)
      print('value encoder: signature')
    elif args.value_encoder == 'char_sig':
      val = CharAndPropSigValueEncoder(value_table, hidden_size=args.embed_dim)
      print('value encoder: char+sig')
    elif args.value_encoder == 'bustle_sig':
      val = BustlePropSigValueEncoder(hidden_size=args.embed_dim)
      print('value encoder: bustle signature')
    else:
      raise ValueError('unknown value encoder %s' % args.value_encoder)
    if args.encode_weight:
      self.encode_weight = ValueWeightEncoder(hidden_size=args.embed_dim)
    else:
      self.encode_weight = DummyWeightEncoder()    
    self.val = val
    arg_mod = LSTMArgSelector
    init_mod = OpPoolingState
    self.init = init_mod(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')
    self.arg = arg_mod(hidden_size=args.embed_dim,
                       mlp_sizes=[256, 1],
                       step_score_func=args.step_score_func,
                       step_score_normalize=args.score_normed)

  def batch_init(self, io_embed, io_scatter, val_embed, value_indices, operation, sample_indices=None, io_gather=None):
    return self.init.batch_forward(io_embed, io_scatter, val_embed, value_indices, operation, sample_indices, io_gather)


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
    if args.encode_weight:
      self.encode_weight = ValueWeightEncoder(hidden_size=args.embed_dim)
    else:
      self.encode_weight = DummyWeightEncoder()