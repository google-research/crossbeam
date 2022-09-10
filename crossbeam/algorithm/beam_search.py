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
import functools
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from crossbeam.algorithm.variables import MAX_NUM_ARGVS
from crossbeam.dsl import value as value_module

N_INF = -1e10
EPS = 1e-8


def _beam_step(score_model, k, cur_state, choice_embed, prefix_scores, choice_mask=None, is_stochastic=False):
    scores = score_model.step_score(cur_state, choice_embed)  # result in a score matrix of size (N-state, N-choice)
    joint_scores = prefix_scores.unsqueeze(1) + scores # broadcast over columns    
    num_choices = choice_embed.shape[0]
    if choice_mask is not None:
      joint_scores = joint_scores * choice_mask + (1 - choice_mask) * N_INF
    joint_scores = joint_scores.view(-1)
    cur_k = joint_scores.shape[0] if k > joint_scores.shape[0] else k
    if is_stochastic:
      prob = torch.softmax(joint_scores, dim=0)
      arg_selected = torch.multinomial(prob, cur_k)
      prefix_scores = joint_scores[arg_selected]
      prefix_scores, idx_sorted = torch.sort(prefix_scores)
      arg_selected = arg_selected[idx_sorted]
    else:
      prefix_scores, arg_selected = torch.topk(joint_scores, cur_k)
    prev_index = torch.div(arg_selected, num_choices, rounding_mode='floor')
    op_choice = arg_selected % num_choices
    prev_state = score_model.state_select(cur_state, prev_index)
    cur_op_embed = choice_embed[op_choice]
    cur_state = score_model.step_state(prev_state, cur_op_embed)
    return cur_state, prev_index, op_choice, prefix_scores


def beam_search(arity, k, values, value_embed, special_var_embed, init_embed, score_model, device, choice_masks=None, is_stochastic=False):
  """
  Args:
    arity: int, arity of current op;
    k: beam_size;
    values: list of values;
    value_embed: array of size [N, embed_dim], embedding of N choices;    
    special_var_embed: embedding of freevar+bondvar
    init_embed: embedding of initial state    
    score_model: an nn module that implements fn_init_state, fn_step_state, fn_step_score
      fn_init_state: initialize score_model state with init_embed
      fn_step_state: a function takes (state, update_embed) as input,
        and output new state; Note that parallel update for beam_size > 1 is done with jax.vmap;
        so fn_step_state only takes care of single state update
      fn_step_score: a function takes (N-state, M-choices) and return a score matrix of N x M
    device: device to run beam search
    choice_mask: list of vector of size N, mark valid (1) or invalid choice(0)
    is_stochastic: whether use stochastic (multinomial) instead of top-k
  Returns:
    arg_choices: jax int32 array of size k x arity, the indices of selected args
    prefix_scores: scores for each arg-list
  """
  cur_state = score_model.get_init_state(init_embed, batch_size=1)
  prefix_scores = torch.zeros(1).to(device)
  arg_choices = torch.LongTensor([[]]).to(device)

  # select args
  for step in range(arity):
    if choice_masks is not None and len(choice_masks):
      choice_mask = choice_masks[step][1].view(1, -1).float()
    else:
      choice_mask = None
    cur_state, prev_index, op_choice, prefix_scores = _beam_step(score_model, k, cur_state,
                                                                 value_embed, prefix_scores,
                                                                 choice_mask=choice_mask,
                                                                 is_stochastic=is_stochastic)
    new_arg_choices = arg_choices[prev_index]
    new_arg_choices = torch.cat((new_arg_choices, op_choice.unsqueeze(1)), axis=1)
    arg_choices = new_arg_choices

  # bind free_var/bond_var for each arg
  for step in range(arity):
    for inner_step in range(MAX_NUM_ARGVS):
      cur_args = arg_choices[:, step].detach().cpu().numpy()
      num_arg_vars = []
      for i in cur_args:
        arg = values[i]
        num_var = 0 if isinstance(arg, value_module.FreeVariable) else arg.num_free_variables
        num_arg_vars.append(num_var)
      stop_indices = torch.LongTensor([i for i, x in enumerate(num_arg_vars) if x <= inner_step]).to(device)
      cur_beam_size = k - stop_indices.shape[0]

      if cur_beam_size == 0:
        invalid_args = torch.zeros(arg_choices.shape[0], MAX_NUM_ARGVS - inner_step).to(arg_choices) - 1
        arg_choices = torch.cat((arg_choices, invalid_args), dim=1)
        break

      step_indices = torch.LongTensor([i for i, x in enumerate(num_arg_vars) if x > inner_step]).to(device)      
      sub_cur_state = score_model.state_select(cur_state, step_indices)
      sub_prefix_scores = prefix_scores[step_indices]
      sub_new_state, sub_prev_idx, sub_op_choices, sub_prefix_scores = _beam_step(score_model, cur_beam_size,
                                                                                  sub_cur_state, special_var_embed,
                                                                                  sub_prefix_scores, is_stochastic=is_stochastic)
      prefix_scores = torch.cat([sub_prefix_scores, prefix_scores[stop_indices]], dim=0)
      sub_arg_choices = arg_choices[step_indices]
      arg_choices = torch.cat([sub_arg_choices[sub_prev_idx], arg_choices[stop_indices]], dim=0)

      cur_state = score_model.state_concat([sub_new_state, score_model.state_select(cur_state, stop_indices)])
      invalid_args = torch.zeros(stop_indices.shape[0], 1).to(arg_choices) - 1
      step_argvars = torch.cat([sub_op_choices.view(-1, 1), invalid_args], dim=0)
      arg_choices = torch.cat([arg_choices, step_argvars], dim=1)

  assert arg_choices.shape[1] == arity + MAX_NUM_ARGVS * arity
  return arg_choices


def beam_step(raw_scores, cur_sizes, beam_size):
  pad_size = max(cur_sizes)
  batch_size = len(cur_sizes)
  n_choices = raw_scores.shape[1]
  if pad_size != min(cur_sizes):
    raw_scores = raw_scores.split(cur_sizes, dim=0)
    padded_scores = pad_sequence(raw_scores, batch_first=True, padding_value=N_INF)
  else:
    padded_scores = raw_scores
  padded_scores = padded_scores.view(batch_size, -1)
  topk_scores, candidates = padded_scores.topk(min(beam_size, padded_scores.shape[1]), dim=-1, sorted=True)
  pred_opts = candidates % n_choices
  pos_index = []
  gap = 0
  for i, s in enumerate(cur_sizes):
    pos_index.append(i * pad_size - gap)
    gap += pad_size - s
  pos_index = torch.LongTensor(pos_index).to(padded_scores.device).view(-1, 1)
  predecessors = torch.div(candidates, n_choices, rounding_mode='floor') + pos_index.expand_as(candidates)

  valid = topk_scores > EPS + N_INF
  n_valid = valid.sum(dim=-1)
  cur_sizes = n_valid.data.cpu().numpy().tolist()
  predecessors = predecessors[valid]
  pred_opts = pred_opts[valid]
  scores = topk_scores[valid].view(-1, 1)
  return predecessors, pred_opts, scores, cur_sizes
