import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import functools
from typing import Sequence
import numpy as np


class EncoderLSTM(nn.Module):
  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.LSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(hidden_size):
    # use dummy key since default state init fn is just zeros.
    return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (), hidden_size)


class CharSeqEncoder(nn.Module):
  vocab_size: int
  hidden_size: int

  @nn.compact
  def __call__(self, input_chars):
    encoder_lstm = EncoderLSTM()
    init_carry = encoder_lstm.initialize_carry(self.hidden_size)
    (_, h), _ = encoder_lstm(init_carry, input_chars)
    return h


class MLP(nn.Module):
  sizes: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for size in self.sizes[:-1]:
        x = nn.Dense(size)(x)
        x = nn.relu(x)
    return nn.Dense(self.sizes[-1])(x)
