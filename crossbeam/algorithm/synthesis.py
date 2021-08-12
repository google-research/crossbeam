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

import numpy as np
import random
import timeit

import torch

from crossbeam.algorithm.beam_search import beam_search
from crossbeam.dsl import value as value_module
from crossbeam.unique_randomizer import unique_randomizer as ur

NUM_VALUES_EXPLORED = 0

def synthesize(task, domain, model, device,
               trace=None, max_weight=15, k=2, is_training=False,
               include_as_train=None, timeout=None, is_stochastic=False,
               random_beam=False, use_ur=False):
  global NUM_VALUES_EXPLORED
  NUM_VALUES_EXPLORED = 0

  verbose = False
  end_time = None if timeout is None or timeout < 0 else timeit.default_timer() + timeout
  if trace is None:
    trace = []
  if include_as_train is None:
    include_as_train = lambda trace_in_beam: True
  num_examples = task.num_examples

  all_values = []

  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  for constant in constants_extractor(task):
    all_values.append(value_module.ConstantValue(constant,
                                                 num_examples=num_examples))
  for input_name, input_value in task.inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, name=input_name))
  output_value = value_module.OutputValue(task.outputs)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  if not random_beam:
    io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
  training_samples = []

  while True:
    cur_num_values = len(all_values)

    for operation in domain.operations:
      if end_time is not None and timeit.default_timer() > end_time:
        return None, all_values
      if verbose:
        print('Operation: {}'.format(operation))
      num_values_before_op = len(all_values)

      if operation.arg_types() is not None:
        type_masks = []
        for arg_index in range(operation.arity):
          arg_type = operation.arg_types()[arg_index]
          type_mask = torch.BoolTensor([v.type == arg_type for v in all_values])
          type_masks.append(type_mask)
        if any(not any(type_mask) for type_mask in type_masks):
          continue  # No options for some argument!
      else:
        type_masks = None

      if use_ur:
        assert not is_training
        randomizer = ur.UniqueRandomizer()
        val_embed = model.val(all_values, device=device)
        init_embed = model.init(io_embed, val_embed, operation)

        new_values = []
        score_model = model.arg
        num_tries = 0
        init_state = score_model.get_init_state(init_embed, batch_size=1)
        randomizer.current_node.cache['state'] = init_state
        while len(new_values) < k and num_tries < 10*k and not randomizer.exhausted():
          num_tries += 1
          arg_list = []
          for arg_index in range(operation.arity):
            cur_state = randomizer.current_node.cache['state']
            if randomizer.needs_probabilities():
              scores = score_model.step_score(cur_state, val_embed)
              scores = scores.view(-1)
              if type_masks is not None:
                scores = torch.where(type_masks[arg_index], scores,
                                     torch.FloatTensor([-1e10]))
              prob = torch.softmax(scores, dim=0)
            else:
              prob = None
            choice_index = randomizer.sample_distribution(prob)
            arg_list.append(all_values[choice_index])
            if 'state' not in randomizer.current_node.cache:
              choice_embed = val_embed[[choice_index]]
              if cur_state is None:
                raise ValueError('cur_state is None!!')
              cur_state = score_model.step_state(cur_state, choice_embed)
              randomizer.current_node.cache['state'] = cur_state
          randomizer.mark_sequence_complete()

          result_value = operation.apply(arg_list)
          NUM_VALUES_EXPLORED += 1
          if verbose and result_value is None:
            print('Cannot apply {} to {}'.format(operation, arg_list))
          if result_value is None or result_value.weight > max_weight:
            continue
          if (domain.small_value_filter and
              not all(domain.small_value_filter(v) for v in result_value.values)):
            continue
          if result_value in all_value_dict:
            # TODO: replace existing one if this way is simpler (less weight).
            # This also means using the simplest forms when recursively
            # reconstructing expressions
            if verbose:
              print('duplicate value: {}, {}'.format(result_value, result_value.expression()))
            continue
          if verbose:
            print('new value: {}, {}'.format(result_value, result_value.expression()))
          new_values.append(result_value)

        for new_value in new_values:
          all_value_dict[new_value] = len(all_values)
          all_values.append(new_value)
          if new_value == output_value:
            return new_value, all_values

        continue

      if random_beam:
        args = [[random.randrange(len(all_values)) for _ in range(operation.arity)]
                for _ in range(k)]
      else:
        val_embed = model.val(all_values, device=device)
        op_state = model.init(io_embed, val_embed, operation)
        args, _ = beam_search(operation.arity, k,
                              val_embed,
                              op_state,
                              model.arg,
                              device=device,
                              is_stochastic=is_stochastic)
        args = args.data.cpu().numpy().astype(np.int32)
      if k > (len(all_values) ** operation.arity):
        args = args[:len(all_values) ** operation.arity]
      beam = [[all_values[i] for i in arg_list] for arg_list in args]

      trace_in_beam = -1
      for i, arg_list in enumerate(beam):
        result_value = operation.apply(arg_list)
        if result_value is None or result_value.weight > max_weight:
          continue
        if (domain.small_value_filter and
            not all(domain.small_value_filter(v) for v in result_value.values)):
          continue
        if result_value in all_value_dict:
          # TODO: replace existing one if this way is simpler (less weight)
          continue
        all_value_dict[result_value] = len(all_values)
        all_values.append(result_value)
        if result_value == output_value and not is_training:
          return result_value, all_values
        # TODO: allow multi-choice when options in trace have the same priority
        # one easy fix would to include this into trace_generation stage (add stochasticity)
        if len(trace) and result_value == trace[0] and trace_in_beam < 0:
          trace_in_beam = i
      if is_training and len(trace) and trace[0].operation == operation:
        if include_as_train(trace_in_beam):  # construct training example
          if trace_in_beam < 0:  # true arg not found
            true_args = []
            true_val = trace[0]
            if not true_val in all_value_dict:
              all_value_dict[true_val] = len(all_values)
              all_values.append(true_val)
            true_arg_vals = true_val.arg_values
            for i in range(operation.arity):
              assert true_arg_vals[i] in all_value_dict
              true_args.append(all_value_dict[true_arg_vals[i]])
            true_args = np.array(true_args, dtype=np.int32)
            args = np.concatenate((args, np.expand_dims(true_args, 0)), axis=0)
            trace_in_beam = args.shape[0] - 1
          training_samples.append((args, trace_in_beam, num_values_before_op, operation))
        trace.pop(0)
        if len(trace) == 0:
          return training_samples, all_values
    if len(all_values) == cur_num_values:  # no improvement
      break
  return None, all_values
