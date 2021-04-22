import torch
import functools
import math
from crossbeam.model.util import ceil_power_of_2


def beam_search(arity, k, choice_embed, init_embed, score_model, choice_mask=None):
  """
  Args:
    arity: int, arity of current op;
    k: beam_size;
    choice_embed: array of size [N, embed_dim], embedding of N choices;
    choice_mask: vector of size N, mark valid (1) or invalid choice(0)
    init_embed: embedding of initial state
    params: FrozenDict of nn parameters, used by score_model
    score_model: an nn module that implements fn_init_state, fn_step_state, fn_step_score
      fn_init_state: initialize score_model state with init_embed
      fn_step_state: a function takes (state, update_embed) as input,
        and output new state; Note that parallel update for beam_size > 1 is done with jax.vmap;
        so fn_step_state only takes care of single state update
      fn_step_score: a function takes (N-state, M-choices) and return a score matrix of N x M

  Returns:
    arg_choices: jax int32 array of size k x arity, the indices of selected args
    prefix_scores: scores for each arg-list
  """
  cur_state = score_model.get_init_state(init_embed, batch_size=1)
  num_choices = choice_embed.shape[0]
  prefix_scores = torch.zeros(1).to(score_model.device)
  arg_choices = torch.LongTensor([[]]).to(score_model.device)
  if choice_mask is not None:
    choice_mask = choice_mask.unsqueeze(0)
  for _ in range(arity):
    scores = score_model.step_score(cur_state, choice_embed)  # result in a score matrix of size (N-state, N-choice)
    joint_scores = prefix_scores.unsqueeze(1) + scores # broadcast over columns
    if choice_mask is not None:
      joint_scores = joint_scores * choice_mask + (1 - choice_mask) * -1e10
    joint_scores = joint_scores.view(-1)
    prefix_scores, arg_topk = torch.topk(joint_scores, joint_scores.shape[0] if k > joint_scores.shape[0] else k)
    prev_index = arg_topk // num_choices
    op_choice = arg_topk % num_choices

    prev_state = score_model.state_select(cur_state, prev_index)
    cur_op_embed = choice_embed[op_choice]
    cur_state = score_model.step_state(prev_state, cur_op_embed)

    new_arg_choices = arg_choices[prev_index]
    new_arg_choices = torch.cat((new_arg_choices, op_choice.unsqueeze(1)), axis=1)
    arg_choices = new_arg_choices
  return arg_choices, prefix_scores
