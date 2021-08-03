import torch
import functools
import math
from crossbeam.model.util import ceil_power_of_2
from torch.nn.utils.rnn import pad_sequence
import numpy as np

N_INF = -1e10
EPS = 1e-8


def beam_search(beam_steps, k, choice_embed, init_embed, score_model, device, num_op_candidates=0, choice_mask=None, is_stochastic=False):
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
    num_op_candidates: number of op candidates to select, 0 if not needed in beam search
    choice_mask: vector of size N, mark valid (1) or invalid choice(0)
    is_stochastic: whether use stochastic (multinomial) instead of top-k
  Returns:
    arg_choices: jax int32 array of size k x arity, the indices of selected args
    prefix_scores: scores for each arg-list
  """
  cur_state = score_model.get_init_state(init_embed, batch_size=1)
  num_choices = choice_embed.shape[0]
  prefix_scores = torch.zeros(1).to(device)
  arg_choices = torch.LongTensor([[]]).to(device)

  if choice_mask is None and num_op_candidates:
    choice_mask = torch.zeros(num_choices).to(device)
    choice_mask[:num_op_candidates] = 1.0  
  if choice_mask is not None:
    choice_mask = choice_mask.unsqueeze(0)
  for step in range(beam_steps):
    scores = score_model.step_score(cur_state, choice_embed)  # result in a score matrix of size (N-state, N-choice)
    joint_scores = prefix_scores.unsqueeze(1) + scores # broadcast over columns    
    if choice_mask is not None:
      joint_scores = joint_scores * choice_mask + (1 - choice_mask) * N_INF
    if step == 0 and num_op_candidates:
      choice_mask = 1.0 - choice_mask    
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
    prev_index = arg_selected // num_choices
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
  predecessors = candidates // n_choices + pos_index.expand_as(candidates)

  valid = topk_scores > EPS + N_INF
  n_valid = valid.sum(dim=-1)
  cur_sizes = n_valid.data.cpu().numpy().tolist()
  predecessors = predecessors[valid]
  pred_opts = pred_opts[valid]
  scores = topk_scores[valid].view(-1, 1)
  return predecessors, pred_opts, scores, cur_sizes


def batch_beam_search(beam_steps, k, choice_embed, choice_indices, init_embed, score_model, device, is_stochastic=False):
  batch_size = len(choice_indices)
  cur_state = score_model.get_batch_init_state(init_embed)
  arg_choices = torch.LongTensor([[] for _ in range(batch_size)]).to(device)
  prefix_scores = torch.zeros(batch_size, 1).to(device)
  ancestors = torch.LongTensor(list(range(batch_size))).to(device)
  cur_sizes = [1] * batch_size

  mask = torch.zeros(batch_size, choice_embed.shape[0]).to(device)
  for i, idx in enumerate(choice_indices):
    mask[i, idx] = 1.0
  for _ in range(beam_steps):
    scores = score_model.step_score(cur_state, choice_embed)
    joint_scores = prefix_scores + scores
    joint_scores = joint_scores * mask + (1 - mask) * N_INF

    predecessors, op_choice, prefix_scores, cur_sizes = beam_step(joint_scores, cur_sizes, k)
    ancestors = ancestors[predecessors]
    mask = mask[predecessors]
    prev_state = score_model.state_select(cur_state, predecessors)
    cur_op_embed = choice_embed[op_choice]
    cur_state = score_model.step_state(prev_state, cur_op_embed)

    new_arg_choices = arg_choices[predecessors]
    arg_choices = torch.cat((new_arg_choices, op_choice.unsqueeze(1)), axis=1)
  arg_choices = arg_choices.data.cpu().split(cur_sizes, dim=0)
  beam_batch = [args.numpy().astype(np.int32) for args in arg_choices]
  return beam_batch, prefix_scores