import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools

from crossbeam.model.base import MLP, DeviceMod


class MLPStepScore(nn.Module):
  def __init__(self, sizes, step_score_normalize: bool):
    super(MLPStepScore, self).__init__()
    self.mlp = MLP(sizes[0], sizes[1:])
    self.step_score_normalize = step_score_normalize

  def forward(self, state, x, is_outer):
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


class LSTMArgSelector(DeviceMod):
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

  def get_init_state(self, init_state, batch_size):
    init_state = init_state.view(1, 1, self.hidden_size)
    h0 = init_state.repeat([self.n_lstm_layers, batch_size, 1])
    return (h0, h0)

  def step_score(self, state, x):
    h, _ = state
    return self.step_func_mod(h[-1], x, is_outer=True)

  def step_state(self, state, x):
    assert len(x.shape) == 2
    x = x.unsqueeze(1)
    return self.lstm(x, state)[1]

  def forward(self, init_state, choice_embed, arg_seq):
    h0, c0 = self.get_init_state(init_state, batch_size=arg_seq.shape[0])
    arg_seq_embed = choice_embed[arg_seq]
    output, _ = self.lstm(arg_seq_embed, (h0, c0))
    state = torch.cat((h0[-1].unsqueeze(1), output[:, :-1, :]), dim=1)
    state = state.view(-1, state.shape[-1])

    if self.step_score_normalize:
      step_logits = self.step_func_mod(state, choice_embed, is_outer=True)
      step_scores = step_logits[torch.arange(state.shape[0]), arg_seq.view(-1)]
    else:
      step_scores = self.step_func_mod(state, arg_seq_embed.view(state.shape), is_outer=False)
    step_scores = step_scores.view(-1, arg_seq.shape[1])
    return step_scores
