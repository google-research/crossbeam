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
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools

from crossbeam.model.base import MLP


class MLPStepScore(nn.Module):
  def __init__(self, sizes, step_score_normalize: bool):
    super(MLPStepScore, self).__init__()
    self.mlp = MLP(sizes[0], sizes[1:])
    self.step_score_normalize = step_score_normalize

  def forward(self, state, x, is_outer, masks=None):
    h = state
    if is_outer:
      num_states = h.shape[0]
      num_x = x.shape[0]
      h = h.repeat_interleave(num_x, dim=0)
      x = x.repeat(num_states, 1)
    else:
      assert state.shape[0] == x.shape[0]
    x = torch.cat((h, x), dim=-1)
    score = self.mlp(x)
    if is_outer:
      score = score.view(num_states, num_x)
      if masks is not None:
        score = score * masks + (1 - masks) * -1e10
      if self.step_score_normalize:
        score = F.log_softmax(score, dim=-1)
    return score


class InnerprodStepScore(nn.Module):
  def __init__(self, step_score_normalize: bool):
    super(InnerprodStepScore, self).__init__()
    self.step_score_normalize = step_score_normalize

  def forward(self, state, x):
    score = torch.matmul(state, x.T)
    if self.step_score_normalize:
      score = F.log_softmax(score, dim=-1)
    return score


class LSTMArgSelector(nn.Module):
  def __init__(self, hidden_size, mlp_sizes, n_lstm_layers = 1,
               step_score_func: str = 'mlp', step_score_normalize: bool = False):
    super(LSTMArgSelector, self).__init__()
    self.hidden_size = hidden_size
    self.mlp_sizes = mlp_sizes
    self.n_lstm_layers = n_lstm_layers
    self.step_score_normalize = step_score_normalize
    self.lstm = nn.LSTM(hidden_size, hidden_size, self.n_lstm_layers, 
                        bidirectional=False, batch_first=True)
    if step_score_func == 'mlp':
      self.step_func_mod = MLPStepScore([self.hidden_size * 2] + mlp_sizes, step_score_normalize)
    elif step_score_func == 'innerprod':
      self.step_func_mod = InnerprodStepScore(step_score_normalize)

  def state_select(self, state, indices, axis=1):
    assert axis == 1
    return (state[0][:, indices, :], state[1][:, indices, :])

  def state_concat(self, states):
    c, h = zip(*states)
    return (torch.cat(c, dim=1), torch.cat(h, dim=1))

  def get_batch_init_state(self, init_state):
    h0 = init_state.unsqueeze(0).repeat(self.n_lstm_layers, 1, 1)
    return (h0, h0)

  def get_init_state(self, init_state, batch_size):
    init_state = init_state.view(1, 1, self.hidden_size)
    h0 = init_state.repeat([self.n_lstm_layers, batch_size, 1])
    return (h0, h0)

  def step_score(self, state, x):
    h, _ = state
    return self.step_func_mod(h[-1], x, is_outer=True)

  def step_state(self, state, choice_embed, x):
    assert len(x.shape) == 2
    x = x.unsqueeze(1)
    if x.device == torch.device('cpu'):
      new_state = self.lstm(x, state)[1]
    else:
      with torch.cuda.device(x.device):
        new_state = self.lstm(x, state)[1]
    return new_state

  def get_step_scores(self, h0, c0, choice_embed, arg_seq, masks=None, need_last_state=False):
    arg_seq_embed = choice_embed[arg_seq]
    if h0.device == torch.device('cpu'):
      output, last_state = self.lstm(arg_seq_embed, (h0, c0))
    else:
      with torch.cuda.device(h0.device):
        output, last_state = self.lstm(arg_seq_embed, (h0, c0))
    state = torch.cat((h0[-1].unsqueeze(1), output[:, :-1, :]), dim=1)
    if masks is not None:
      masks = masks.unsqueeze(1).repeat(1, state.shape[1], 1).view(-1, masks.shape[-1])
    state = state.view(-1, state.shape[-1])

    if self.step_score_normalize:
      step_logits = self.step_func_mod(state, choice_embed, is_outer=True, masks=masks)
      step_scores = step_logits[torch.arange(state.shape[0]), arg_seq.view(-1)]
    else:
      step_scores = self.step_func_mod(state, arg_seq_embed.view(state.shape), is_outer=False)
    step_scores = step_scores.view(-1, arg_seq.shape[1])
    if need_last_state:
      return step_scores, last_state
    return step_scores

  def forward(self, init_state, choice_embed, arg_seq, masks=None, need_last_state=False):
    if not isinstance(init_state, tuple):
      h0, c0 = self.get_init_state(init_state, batch_size=arg_seq.shape[0])
    else:
      h0, c0 = init_state
    return self.get_step_scores(h0, c0, choice_embed, arg_seq, masks=masks, need_last_state=need_last_state)

  def batch_forward(self, init_state, choice_embed, arg_seq, masks=None):
    h0, c0 = self.get_batch_init_state(init_state)
    step_scores = self.get_step_scores(h0, c0, choice_embed, arg_seq, masks)
    return step_scores


class AttnLstmArgSelector(LSTMArgSelector):
  def __init__(self, hidden_size, mlp_sizes, n_lstm_layers = 1,
               step_score_func: str = 'mlp', step_score_normalize: bool = False):
    super(AttnLstmArgSelector, self).__init__(
      hidden_size, mlp_sizes, n_lstm_layers, step_score_func, step_score_normalize
    )
    if step_score_func == 'mlp':
      self.attn_func_mod = MLPStepScore([self.hidden_size * 2] + mlp_sizes, step_score_normalize)
    elif step_score_func == 'innerprod':
      self.attn_func_mod = InnerprodStepScore(step_score_normalize)
    self.state_input = MLP(hidden_size * 2, [hidden_size * 2, hidden_size])

  def step_state(self, state, choice_embed, x):
    assert len(x.shape) == 2
    h, _ = state
    attn_logits = self.attn_func_mod(h[-1], choice_embed, is_outer=True)
    attn = F.softmax(attn_logits, dim=-1)

    context = torch.matmul(attn, choice_embed)
    x = torch.cat([context, x], axis=-1)
    x = self.state_input(x)
    x = x.unsqueeze(1)

    if x.device == torch.device('cpu'):
      new_state = self.lstm(x, state)[1]
    else:
      with torch.cuda.device(x.device):
        new_state = self.lstm(x, state)[1]
    return new_state

  def get_step_scores(self, h0, c0, choice_embed, arg_seq, masks=None, need_last_state=False):
    arg_seq_embed = choice_embed[arg_seq]
    state = (h0, c0)
    list_h = [h0[-1].unsqueeze(1)]
    for step in range(arg_seq_embed.shape[1]):
      cur_arg_embed = arg_seq_embed[:, step]
      state = self.step_state(state, choice_embed, cur_arg_embed)
      h, _ = state
      list_h.append(h[-1].unsqueeze(1))
    last_state = state
    state = torch.cat(list_h[:-1], dim=1)
    if masks is not None:
      masks = masks.unsqueeze(1).repeat(1, state.shape[1], 1).view(-1, masks.shape[-1])
    state = state.view(-1, state.shape[-1])

    if self.step_score_normalize:
      step_logits = self.step_func_mod(state, choice_embed, is_outer=True, masks=masks)
      step_scores = step_logits[torch.arange(state.shape[0]), arg_seq.view(-1)]
    else:
      step_scores = self.step_func_mod(state, arg_seq_embed.view(state.shape), is_outer=False)
    step_scores = step_scores.view(-1, arg_seq.shape[1])
    if need_last_state:
      return step_scores, last_state
    return step_scores

