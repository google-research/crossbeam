from crossbeam.datasets import random_data
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools

from crossbeam.model.util import CharacterTable
from crossbeam.model.encoder import CharIOLSTMEncoder, CharValueLSTMEncoder
from crossbeam.model.op_init import OpPoolingState
from crossbeam.model.op_arg import LSTMArgSelector


if __name__ == '__main__':

  embed_dim = 128
  operations = tuple_operations.get_operations()
  def task_gen(constants, operations):
    while True:
      task = random_data.generate_random_task(
          min_weight=3,
          max_weight=6,
          num_examples=2,
          num_inputs=3,
          constants=constants,
          operations=operations,
          input_generator=random_data.RANDOM_INTEGER)  
      if task:
        return task
  constants = [0]
  t = task_gen(constants, operations)
  
  input_table = CharacterTable('0123456789:,', max_len=50)
  output_table = CharacterTable('0123456789() ,', max_len=50)
  value_table = CharacterTable('0123456789intuple:[]() ,', max_len=70)
  model = CharIOLSTMEncoder(input_table, output_table, hidden_size=embed_dim)
  io_embed = model(t.inputs_dict, t.outputs)
  print(io_embed.shape)

  all_values = []
  for constant in constants:
    all_values.append(value_module.ConstantValue(constant,
                                                 num_examples=2))
  for input_name, input_value in t.inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, name=input_name))
  
  model = CharValueLSTMEncoder(value_table, hidden_size=embed_dim)
  value_embed = model(all_values)
  print(value_embed.shape)
  
  model = OpPoolingState(operations, embed_dim, 'max')

  state = model(io_embed, value_embed, operations[0])

  model = LSTMArgSelector(embed_dim, [256, 1])
  arg_sel = torch.LongTensor([[0, 1, 2], [1, 2, 3]])
  out = model(state, value_embed, arg_sel)

  init_state = model.get_init_state(state, batch_size=2)  
  print(init_state[0].shape)
  print(model.step_score(init_state, value_embed).shape)
  x = value_embed[[1, 2]]
  new_state = model.step_state(init_state, x)
  print(new_state[0].shape)
