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

N_INF = -1e10
EPS = 1e-8


def beam_search(beam_steps, k, choice_embed, init_embed, score_model, device, choice_masks=None, is_stochastic=False):
  """
  Args:
    beam_steps: int, beam_steps to perform;
    k: beam_size;
    choice_embed: array of size [N, embed_dim], embedding of N choices;    
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
  num_choices = choice_embed.shape[0]
  prefix_scores = torch.zeros(1).to(device)
  arg_choices = torch.LongTensor([[]]).to(device)
  for step in range(beam_steps):
    scores = score_model.step_score(cur_state, choice_embed)  # result in a score matrix of size (N-state, N-choice)
    joint_scores = prefix_scores.unsqueeze(1) + scores # broadcast over columns    
    if choice_masks is not None and len(choice_masks):
      choice_mask = choice_masks[step][1].view(1, -1).float()
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

    new_arg_choices = arg_choices[prev_index]
    new_arg_choices = torch.cat((new_arg_choices, op_choice.unsqueeze(1)), axis=1)
    arg_choices = new_arg_choices
  return arg_choices, prefix_scores


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
