import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import functools
from typing import Sequence

from crossbeam.model.base import MLP


class PoolingState(nn.Module):
  state_dim: int
  pool_method: str = 'mean'

  @nn.compact
  def __call__(self, io_embed, value_embed, value_mask):
    pool_method = getattr(jnp, self.pool_method, None)
    io_state = pool_method(io_embed, axis=0, keepdims=True)
    value_mask = jnp.expand_dims(value_mask, 1)
    if self.pool_method == 'mean':
      value_state = jnp.sum(value_embed * value_mask, axis=0, keepdims=True)
      value_state = value_state / (jnp.sum(value_mask) + 1e-10)
    elif self.pool_method == 'max':
      value_state = value_embed * value_mask + -1e10 * (1 - value_mask)
      value_state = jnp.max(value_state, axis=0, keepdims=True)
    else:
      raise NotImplementedError

    joint_state = jnp.concatenate((io_state, value_state), axis=1)
    proj = nn.Dense(self.state_dim)
    return proj(joint_state)


class OpPoolingState(nn.Module):  
  ops: Sequence
  state_dim: int
  pool_method: str = 'mean'

  def setup(self):
    self.op_specific_mod = [PoolingState(self.state_dim) for _ in range(len(self.ops))]
    self.op_idx_map = {repr(op): i for i, op in enumerate(self.ops)}

  def __call__(self, io_embed, value_embed, value_mask, op):
    if op is None:
      joint_state = []
      for op in self.op_idx_map:
        mod = self.op_specific_mod[self.op_idx_map[op]]
        state = mod(io_embed, value_embed, value_mask)
        joint_state.append(state)
      joint_state = jnp.concatenate(joint_state, axis=0)
      return joint_state
    else:
      mod = self.op_specific_mod[self.op_idx_map[repr(op)]]
      return mod(io_embed, value_embed, value_mask)

  @functools.partial(jax.jit, static_argnums=[0, 5])
  def encode(self, params, io_embed, value_embed, value_mask, op):
    return self.apply(params, io_embed, value_embed, value_mask, op)

  def init_params(self, key):
    dummy_io = jnp.zeros((1, 2 * self.state_dim), dtype=jnp.float32)
    dummy_mask = jnp.ones(1, dtype=jnp.int32)
    dummy_val = jnp.zeros((1, self.state_dim), dtype=jnp.float32)
    return self.init(key, dummy_io, dummy_val, dummy_mask, None)
