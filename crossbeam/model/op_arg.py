import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import functools
from typing import Sequence

from crossbeam.model.base import MLP
from flax.core import Scope


class MLPStepScore(nn.Module):
  sizes: Sequence[int]
  step_score_normalize: bool = False

  @nn.compact
  def __call__(self, state, x, is_outer):
    h = state
    if is_outer:
      num_states = h.shape[0]
      num_x = x.shape[0]
      h = jnp.repeat(h, num_x, axis=0)
      x = jnp.tile(x, [num_states, 1])
    else:
      assert state.shape[0] == x.shape[0]
    x = jnp.concatenate((h, x), axis=-1)
    mlp = MLP(self.sizes)
    score = mlp(x)
    if is_outer:
      score = jnp.reshape(score, (num_states, num_x))
      if self.step_score_normalize:
        score = jax.nn.log_softmax(score, axis=-1)
    return score


class InnerprodStepScore(nn.Module):
  step_score_normalize: bool = False

  @nn.compact
  def __call__(self, state, x):
    score = jnp.matmul(state, x.T)
    if self.step_score_normalize:
      score = jax.nn.log_softmax(score, axis=-1)
    return score


class LSTMAutoreg(nn.Module):
  hidden_size: int

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.LSTMCell()(carry, x)

  def init_state(self, batch_size=None):
    # use dummy key since default state init fn is just zeros.
    if batch_size is None:
      return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (), self.hidden_size)
    return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size, ), self.hidden_size)


def get_score_mod(args):
  if args.step_score_func == 'mlp':
    step_func_mod = MLPStepScore(args.mlp_sizes, args.step_score_normalize)
  elif args.step_score_func == 'innerprod':
    step_func_mod = InnerprodStepScore(args.step_score_normalize)
  else:
    raise NotImplementedError
  return step_func_mod


class LSTMArgSelector(nn.Module):
  hidden_size: int
  mlp_sizes: Sequence[int] = (256, 1)
  step_score_func: str = 'mlp'
  step_score_normalize: bool = False

  def setup(self):
    assert not self.step_score_normalize  # otherwise we need to handle mask of choice_embed
    self.lstm_state_mod = LSTMAutoreg(self.hidden_size)
    self.step_func_mod = get_score_mod(self)

  def __call__(self, init_state, choice_embed, arg_seq):
    lstm_state = self.lstm_state_mod.init_state(arg_seq.shape[0])
    lstm_state = jax.tree_map(lambda x: x + init_state, lstm_state)
    arg_seq_embed = choice_embed[arg_seq]
    _, state = jax.vmap(self.lstm_state_mod)(lstm_state, arg_seq_embed)
    state = jnp.concatenate((jnp.expand_dims(lstm_state[1], 1), state[:, :-1, :]), axis=1)

    state = jnp.reshape(state, [-1, state.shape[-1]])
    if self.step_score_normalize:
      step_logits = self.step_func_mod(state, choice_embed, is_outer=True)
      step_scores = step_logits[jnp.arange(state.shape[0]), arg_seq.flatten()]
    else:
      step_scores = self.step_func_mod(state, 
                                      jnp.reshape(arg_seq_embed, [state.shape[0], -1]),
                                      is_outer=False)
    step_scores = jnp.reshape(step_scores, [-1, arg_seq.shape[1]])
    return jnp.sum(step_scores, axis=-1, keepdims=True)

  def fn_init_state(self, params):
    def work(init_embed):
      lstm_state = LSTMAutoreg(self.hidden_size).init_state()
      lstm_state = jax.tree_map(lambda x: x + init_embed, lstm_state)
      return lstm_state
    return work

  def fn_step_score(self, params):
    def work(state, x):
      mod = get_score_mod(self)
      p = params['params']['step_func_mod']
      return mod.apply({'params': p}, state[1], x, is_outer=True)
    return work

  def fn_step_state(self, params):
    def work(state, x):
      lstm_state_mod = LSTMAutoreg(self.hidden_size)
      p = params['params']['lstm_state_mod']
      return lstm_state_mod.apply({'params': p}, state, jnp.expand_dims(x, 0))[0]
    return work

  def init_params(self, key):
    dummy_state = jnp.zeros((2, self.hidden_size), dtype=jnp.float32)
    dummy_args = jnp.zeros((4, self.hidden_size), dtype=jnp.float32)
    dummy_arg_sel = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
    return self.init(key, dummy_state, dummy_args, dummy_arg_sel)


if __name__ == '__main__':
  key = jax.random.PRNGKey(1)
  embed_dim = 16

  model = LSTMArgSelector(hidden_size=embed_dim)
  key, arr_key = jax.random.split(key)
  dummy_state = jnp.ones((2, embed_dim), dtype=jnp.float32)
  dummy_args = jax.random.uniform(arr_key, (4, embed_dim))
  dummy_arg_sel = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
  params = model.init(key, dummy_state, dummy_args, dummy_arg_sel)

  for key in params['params']:
    print(key)
  score = model.apply(params, dummy_state, dummy_args, dummy_arg_sel)
  print(score)

  func_step_score = model.fn_step_score(params)
  s = func_step_score((None, dummy_state), dummy_args)
  # s = mod_bind.step_score((None, dummy_state), dummy_args)
  print(s)
  # print(params['params'].keys())
  # print(state.shape)
