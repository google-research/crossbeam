from typing import Callable
import numpy as np


def onehot(sequence, vocab_size):
  """One-hot encode a single sequence of integers."""
  return jnp.array(
      sequence[:, np.newaxis] == jnp.arange(vocab_size), dtype=jnp.float32)


class CharacterTable(object):
  """Encode/decodes between strings and integer representations."""

  @property
  def pad_id(self):
    return 0

  @property
  def eos_id(self):
    return 1

  @property
  def vocab_size(self):
    return len(self._chars) + 2

  def __init__(self, chars, max_len):
    self._chars = sorted(set(chars))
    self._char_indices = dict(
        (ch, idx + 2) for idx, ch in enumerate(self._chars))
    self._indices_char = dict(
        (idx + 2, ch) for idx, ch in enumerate(self._chars))
    self._indices_char[self.pad_id] = '_'
    self.max_len = max_len + 1  # for eos token

  def encode(self, inputs):
    """Encode from string to list of integers."""
    return np.array(
        [self._char_indices.get(char, self._char_indices['_']) for char in inputs] + [self.eos_id])

  def decode(self, inputs):
    """Decode from list of integers to string."""
    chars = []
    for elem in inputs:
      if elem == self.eos_id:
        break
      chars.append(self._indices_char[elem])
    return ''.join(chars)


def make_onehot_tensor(list_obj, fn_tostring: Callable, vocab: CharacterTable):
  list_tokens = []
  cur_lens = []
  for i in list_obj:
    tokens = vocab.encode(fn_tostring(i))
    cur_lens.append(len(tokens))
    assert vocab.max_len >= len(tokens)
    tokens = np.pad(tokens, [(0, vocab.max_len-len(tokens))], mode='constant')
    list_tokens.append(onehot(tokens, vocab.vocab_size))
  token_tensor = jnp.array(list_tokens, dtype=jnp.float32)
  cur_lens = jnp.array(cur_lens, dtype=jnp.int32)
  return token_tensor, cur_lens


def ceil_power_of_2(x):
  return 1 if x == 0 else 2 ** (x - 1).bit_length()


def pad_power_of_2(x, axis=0):
  orig_dim = x.shape[axis]
  padded_dim = ceil_power_of_2(orig_dim)
  pad_config = [(0, 0)] * len(x.shape)
  pad_config[axis] = (0, padded_dim - orig_dim)
  padded_x = jnp.pad(x, pad_config)
  return padded_x
