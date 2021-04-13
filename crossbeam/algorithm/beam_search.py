import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import functools
import math
from crossbeam.model.util import ceil_power_of_2


@functools.partial(jax.jit, static_argnums=[0, 1, 6])
def beam_search(arity, k, choice_embed, choice_mask, init_embed, params, score_model):
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
  state_update_func = score_model.fn_step_state(params)
  vmap_state_func = jax.vmap(state_update_func)
  init_func = score_model.fn_init_state(params)
  init_state = init_func(init_embed.flatten())
  score_func = score_model.fn_step_score(params)

  cur_state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_state)
  num_choices = choice_embed.shape[0]
  prefix_scores = jnp.zeros((1, ), jnp.float32)
  arg_choices = jnp.array([[]], dtype=jnp.int32)
  choice_mask = jnp.expand_dims(choice_mask, 0)
  for _ in range(arity):
    scores = score_func(cur_state, choice_embed)  # result in a score matrix of size (N-state, N-choice)
    joint_scores = jnp.expand_dims(prefix_scores, 1) + scores # broadcast over columns
    joint_scores = joint_scores * choice_mask + (1 - choice_mask) * -1e10
    joint_scores = joint_scores.flatten()
    prefix_scores, arg_topk = jax.lax.top_k(joint_scores,
                                            joint_scores.shape[0] if k > joint_scores.shape[0] else k)
    prev_index = arg_topk // num_choices
    op_choice = arg_topk % num_choices

    prev_state = jax.tree_map(lambda x: jnp.take(x, prev_index, axis=0), cur_state)
    cur_op_embed = jnp.take(choice_embed, op_choice, axis=0)

    cur_state = vmap_state_func(prev_state, cur_op_embed)

    new_arg_choices = arg_choices[prev_index]
    new_arg_choices = jnp.concatenate((new_arg_choices, jnp.expand_dims(op_choice, 1)), axis=1)
    arg_choices = new_arg_choices
  return arg_choices, prefix_scores


def padded_beam_search(arity, k, choice_embed, init_embed, params, score_model):
  """This one pad the choices into size of power-of-2.

  See beam_search for the args.
  """
  num_choices = choice_embed.shape[0]
  #TODO: Depending on how expensive each compilation is,
  # we might not need buckets for small sizes,
  # e.g., any num_choices <= 128 leads to padded_shape=128
  padded_shape = ceil_power_of_2(num_choices)
  choice_mask = jnp.pad(jnp.ones(num_choices), [(0, padded_shape - num_choices)])
  choice_embed = jnp.pad(choice_embed, [(0, padded_shape - num_choices), (0, 0)])
  return beam_search(arity, k, choice_embed, choice_mask, init_embed, params, score_model)
