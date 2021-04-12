from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import functools

from crossbeam.model.base import CharSeqEncoder
from crossbeam.model.util import CharacterTable, make_onehot_tensor, pad_power_of_2


class CharIOLSTMEncoder(nn.Module):
  input_char_table: CharacterTable
  output_char_table: CharacterTable
  hidden_size: int
  to_string: Callable = repr

  @nn.compact
  def __call__(self, input_seq, output_seq):
    input_embed = CharSeqEncoder(self.input_char_table.vocab_size, self.hidden_size)(input_seq)
    output_embed = CharSeqEncoder(self.output_char_table.vocab_size, self.hidden_size)(output_seq)
    cat_embed = jnp.concatenate((input_embed, output_embed), axis=-1)
    return cat_embed

  @functools.partial(jax.jit, static_argnums=0)
  def exec_encode(self, params, imat, omat, imask, omask):
    @functools.partial(jax.mask, in_shapes=['(n, _)', '(m, _)'], out_shape=f'({2 * self.hidden_size},)')
    def single_encode_fn(seq_i, seq_o):
      return self.apply(params, seq_i, seq_o)
    return jax.vmap(single_encode_fn)([imat, omat], dict(n=imask, m=omask))

  def make_input(self, inputs_dict, outputs):
    list_input = [''] * len(outputs)
    for _, input_value in inputs_dict.items():
      for i in range(len(list_input)):
        list_input[i] += self.to_string(input_value[i]) + ','
    list_output = [self.to_string(x) for x in outputs]
    io_mats = []
    io_masks = []
    dummy_ts = lambda x: x
    for l, tab in [(list_input, self.input_char_table), (list_output, self.output_char_table)]:
      tok_tensor, lens = make_onehot_tensor(l, dummy_ts, tab)
      io_mats.append(tok_tensor)
      io_masks.append(lens)
    imat, omat = io_mats
    imask, omask = io_masks
    return imat, omat, imask, omask

  def encode(self, params, inputs_dict, outputs):
    imat, omat, imask, omask = self.make_input(inputs_dict, outputs)
    return self.exec_encode(params, imat, omat, imask, omask)

  def init_params(self, key):
    dummy_in, _ = make_onehot_tensor([self.input_char_table._chars[0]], lambda x: x, self.input_char_table)
    dummy_out, _ = make_onehot_tensor([self.output_char_table._chars[0]], lambda x: x, self.output_char_table)
    return self.init(key, dummy_in[0], dummy_out[0])


class CharValueLSTMEncoder(nn.Module):
  val_char_table: CharacterTable
  hidden_size: int
  to_string: Callable = repr

  @nn.compact
  def __call__(self, val_seq):
    val_embed = CharSeqEncoder(self.val_char_table.vocab_size, self.hidden_size)(val_seq)
    return val_embed

  @functools.partial(jax.jit, static_argnums=0)
  def exec_encode(self, params, val_tensor, val_lens):
    @functools.partial(jax.mask, in_shapes=('(n, _)',), out_shape=f'({self.hidden_size},)')
    def single_encode_fn(seq_val):
      return self.apply(params, seq_val)
    return jax.vmap(single_encode_fn)((val_tensor,), dict(n=val_lens))

  def make_input(self, all_values):
    val_tensor, len_vals = make_onehot_tensor(all_values, self.to_string, self.val_char_table)
    val_tensor = pad_power_of_2(val_tensor, axis=0)
    len_vals = pad_power_of_2(len_vals, axis=0)
    return val_tensor, len_vals

  def padded_encode(self, params, all_values):
    n_vals = len(all_values)
    val_tensor, len_vals = self.make_input(all_values)
    val_embed = self.exec_encode(params, val_tensor, len_vals)
    pad_mask = jnp.pad(jnp.ones(n_vals), [(0, val_embed.shape[0] - n_vals)])
    return val_embed, pad_mask

  def encode(self, params, all_values):
    n_vals = len(all_values)
    val_embed, _ = self.padded_encode(params, all_values)
    return val_embed[:n_vals]

  def init_params(self, key):
    dummy_seq, _ = make_onehot_tensor([self.val_char_table._chars[0]], lambda x: x, self.val_char_table)
    return self.init(key, dummy_seq[0])
