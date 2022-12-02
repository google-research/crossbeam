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
from crossbeam.model.op_arg import LSTMArgSelector, AttnLstmArgSelector
from crossbeam.model.op_init import PoolingState, OpPoolingState
from crossbeam.model.encoder import LambdaSigIOEncoder
from crossbeam.model.encoder import LambdaSigValueEncoder
from crossbeam.model.encoder import ValueWeightEncoder, DummyWeightEncoder
from crossbeam.algorithm.variables import MAX_NUM_FREE_VARS, MAX_NUM_BOUND_VARS

class DeepCoderModel(nn.Module):
  def __init__(self, args, operations):
    super(DeepCoderModel, self).__init__()
    if args.io_encoder == 'lambda_signature':
      self.io = LambdaSigIOEncoder(args.max_num_inputs, hidden_size=args.embed_dim)
    else:
      raise ValueError('unknown io encoder %s' % args.io_encoder)

    if args.value_encoder == 'lambda_signature':
      val = LambdaSigValueEncoder(hidden_size=args.embed_dim)
    else:
      raise ValueError('unknown value encoder %s' % args.value_encoder)
    if args.encode_weight:
      self.encode_weight = ValueWeightEncoder(hidden_size=args.embed_dim)
    else:
      self.encode_weight = DummyWeightEncoder()
    self.val = val
    if args.arg_selector == 'lstm':
      arg_mod = LSTMArgSelector
    elif args.arg_selector == 'attn_lstm':
      arg_mod = AttnLstmArgSelector
    else:
      raise ValueError('unknown arg selector %s' % args.arg_selector)

    init_mod = OpPoolingState
    self.init = init_mod(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')
    self.arg = arg_mod(hidden_size=args.embed_dim,
                       mlp_sizes=[256, 1],
                       step_score_func=args.step_score_func,
                       step_score_normalize=args.score_normed)

    self.special_var_embed = nn.Parameter(torch.zeros(MAX_NUM_FREE_VARS + MAX_NUM_BOUND_VARS, args.embed_dim))
    nn.init.xavier_uniform_(self.special_var_embed)
