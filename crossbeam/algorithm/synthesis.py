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
from copy import deepcopy
from crossbeam.algorithm.beam_search import beam_search 
from crossbeam.dsl import value as value_module
from crossbeam.unique_randomizer import unique_randomizer as ur

from crossbeam.algorithm.baseline_enumeration import MAX_NUM_FREE_VARS, MAX_NUM_ARGVS
from crossbeam.algorithm.baseline_enumeration import ALL_BOUND_VARS, ALL_FREE_VARS, ARGV_MAP


def init_values(task, domain, all_values):
  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  for constant in constants_extractor(task):
    all_values.append(value_module.ConstantValue(constant))
  for input_name, input_value in task.inputs_dict.items():
    all_values.append(value_module.InputVariable(input_value, name=input_name))
  all_values.extend(ALL_FREE_VARS)
  output_value = value_module.OutputValue(task.outputs)
  return output_value


def update_masks(type_masks, operation, all_values, device, vidx_start=0):
  feasible = True
  for arg_index in range(operation.arity):
    arg_type = operation.arg_types()[arg_index]
    bool_mask = [arg_type is None or all_values[v].type == arg_type for v in range(vidx_start, len(all_values))]
    cur_feasible = any(bool_mask)
    step_type_mask = torch.BoolTensor(bool_mask).to(device)
    if len(type_masks) <= arg_index:
      type_masks.append([cur_feasible, step_type_mask])
    else:
      type_masks[arg_index][1] = torch.cat([type_masks[arg_index][1], step_type_mask])
      cur_feasible = cur_feasible or type_masks[arg_index][0]
      type_masks[arg_index][0] = cur_feasible
    feasible = feasible and cur_feasible
  return feasible


def update_with_better_value(result_value, all_value_dict, all_values, model,
                             device, output_value, verbose):
  old_value = all_values[all_value_dict[result_value]]
  if result_value.get_weight() < old_value.get_weight():
    assert isinstance(old_value, value_module.OperationValue)
    if verbose:
      print('duplicate value found. was: {}, {}, weight {}'.format(
          old_value, old_value.expression(), old_value.get_weight()))
    old_value.operation = result_value.operation
    old_value.arg_values = result_value.arg_values
    if verbose:
      print('  updated to: {}, {}, weight {}'.format(
          old_value, old_value.expression(), old_value.get_weight()))


def copy_operation_value(operation, value, all_values, all_value_dict, trace_values):
  assert isinstance(value, value_module.OperationValue)
  arg_values = []
  for v in value.arg_values:
    if v in all_value_dict:
      arg_values.append(all_values[all_value_dict[v]])
    else:
      arg_values.append(trace_values.index(v))
  if not value.values:
    return operation.apply(arg_values, value.arg_variables, value.free_variables)
  else:
    return value_module.OperationValue(value.values, value.operation, arg_values,
                                       arg_variables=deepcopy(value.arg_variables),
                                       free_variables=deepcopy(value.free_variables))


def decode_args(operation, args, all_values):
  arg_list = args[:operation.arity]
  arg_list = [all_values[x] for x in arg_list]
  offset = operation.arity
  free_vars = set([v for v in arg_list if isinstance(v, value_module.FreeVariable)])
  arg_var_list = []
  for arg in arg_list:
    num_required = (0 if isinstance(arg, value_module.FreeVariable)
                    else arg.num_free_variables)
    cur_arg_vars = []
    for i in range(offset, offset + num_required):
      var_idx = args[i]
      if var_idx >= MAX_NUM_FREE_VARS:  # then it is a bound var
        v = ALL_BOUND_VARS[var_idx - MAX_NUM_FREE_VARS]
      else:
        v = ALL_FREE_VARS[var_idx]
        free_vars.add(v)
      cur_arg_vars.append(v)
    arg_var_list.append(cur_arg_vars)
    offset += MAX_NUM_ARGVS
  assert offset == len(args)
  free_vars = sorted(list(free_vars), key=lambda x: x.name)
  return arg_list, arg_var_list, free_vars


def synthesize(task, domain, model, device,
               trace=None, max_weight=15, k=2, is_training=False,
               include_as_train=None, timeout=None, max_values_explored=None, is_stochastic=False,
               random_beam=False, use_ur=False, masking=True, static_weight=False):
  stats = {'num_values_explored': 0}

  verbose = False
  end_time = None if timeout is None or timeout < 0 else timeit.default_timer() + timeout
  if trace is None:
    trace = []
  trace_values = []
  if include_as_train is None:
    include_as_train = lambda trace_in_beam: True

  all_values = []
  output_value = init_values(task, domain, all_values)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  if not random_beam:
    io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
    val_base_embed = model.val(all_values, device=device, output_values=output_value)
    value_embed = model.encode_weight(val_base_embed, [v.get_weight() for v in all_values])

  training_samples = []
  mask_dict = {}
  for operation in domain.operations:
    type_masks = []
    if operation.arg_types() is not None and masking:
      update_masks(type_masks, operation, all_values, device)
    mask_dict[operation] = type_masks

  while True:
    cur_num_values = len(all_values)
    for operation in domain.operations:
      if (end_time is not None and timeit.default_timer() > end_time) or (
          max_values_explored is not None and stats['num_values_explored'] >= max_values_explored):
        return None, all_values, stats
      if verbose:
        print('Operation: {}'.format(operation))
      num_values_before_op = len(all_values)
      type_masks = mask_dict[operation]

      if len(type_masks):
        if len(all_values) > type_masks[0][1].shape[0]:
          feasible = update_masks(type_masks, operation, all_values, device, vidx_start=type_masks[0][1].shape[0])
          if not feasible:
            continue

      if use_ur:
        assert not is_training
        randomizer = ur.UniqueRandomizer()
        if len(all_values) > val_base_embed.shape[0]:
          more_val_embed = model.val(all_values[val_base_embed.shape[0]:], device=device, output_values=output_value)
          val_base_embed = torch.cat((val_base_embed, more_val_embed), dim=0)
        value_embed = model.encode_weight(val_base_embed, [v.get_weight() for v in all_values])
        init_embed = model.init(io_embed, value_embed, operation)

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
              scores = score_model.step_score(cur_state, value_embed)
              scores = scores.view(-1)
              if len(type_masks):
                scores = torch.where(type_masks[arg_index][1], scores, torch.FloatTensor([-1e10]).to(device))
              prob = torch.softmax(scores, dim=0)
            else:
              prob = None
            choice_index = randomizer.sample_distribution(prob)
            arg_list.append(all_values[choice_index])
            if 'state' not in randomizer.current_node.cache:
              choice_embed = value_embed[[choice_index]]
              if cur_state is None:
                raise ValueError('cur_state is None!!')
              cur_state = score_model.step_state(cur_state, choice_embed)
              randomizer.current_node.cache['state'] = cur_state
          randomizer.mark_sequence_complete()

          result_value = operation.apply(arg_list)
          stats['num_values_explored'] += 1
          if verbose and result_value is None:
            print('Cannot apply {} to {}'.format(operation, arg_list))
          if result_value is None or result_value.get_weight() > max_weight:
            continue
          if (domain.small_value_filter and
              not all(domain.small_value_filter(v) for v in result_value.values)):
            continue
          if result_value in all_value_dict:
            if not static_weight:
              update_with_better_value(result_value, all_value_dict, all_values,
                                       model, device, output_value, verbose)
            continue
          if verbose:
            print('new value: {}, {}'.format(result_value, result_value.expression()))
          new_values.append(result_value)

        for new_value in new_values:
          all_value_dict[new_value] = len(all_values)
          all_values.append(new_value)
          if new_value == output_value:
            return new_value, all_values, stats

        continue
      
      weight_snapshot = [v.get_weight() for v in all_values]
      if random_beam:
        raise NotImplementedError  #TODO(hadai): enable random beam during training
        args = [[] for _ in range(k)]
        for b in range(k):
          args[b] += [np.random.randint(0, len(all_values)) for _ in range(operation.arity)]
      else:
        if len(all_values) > val_base_embed.shape[0]:
          more_val_embed = model.val(all_values[val_base_embed.shape[0]:], device=device, output_values=output_value)
          val_base_embed = torch.cat((val_base_embed, more_val_embed), dim=0)
        value_embed = model.encode_weight(val_base_embed, weight_snapshot)
        op_state = model.init(io_embed, value_embed, operation)
        args = beam_search(operation.arity, k,
                           all_values,
                           value_embed,
                           model.special_var_embed,
                           op_state,
                           model.arg,
                           device=device,
                           choice_masks=type_masks,
                           is_stochastic=is_stochastic)
        args = args.data.cpu().numpy().astype(np.int32)

      trace_in_beam = -1
      for beam_pos, args_and_vars in enumerate(args):
        arg_list, arg_vars, free_vars = decode_args(operation, args_and_vars, all_values)
        result_value = operation.apply(arg_list, arg_vars, free_vars)
        stats['num_values_explored'] += 1
        if result_value is None or result_value.get_weight() > max_weight:
          continue
        if (domain.small_value_filter and
            not all(domain.small_value_filter(v) for v in result_value.values)):
          continue
        if result_value in all_value_dict:
          if not static_weight:
            update_with_better_value(result_value, all_value_dict, all_values,
                                     model, device, output_value, verbose)
          continue
        all_value_dict[result_value] = len(all_values)
        all_values.append(result_value)
        if result_value == output_value and not is_training:
          return result_value, all_values, stats
        # TODO: allow multi-choice when options in trace have the same priority
        # one easy fix would to include this into trace_generation stage (add stochasticity)
        if len(trace) and result_value == trace[0] and trace_in_beam < 0:
          trace_in_beam = beam_pos
      if is_training and len(trace) and trace[0].operation == operation:
        true_val = copy_operation_value(operation, trace[0], all_values, all_value_dict, trace_values)
        if include_as_train(trace_in_beam):  # construct training example
          if trace_in_beam < 0:  # true arg not found
            true_args = []
            if not true_val in all_value_dict:
              all_value_dict[true_val] = len(all_values)
              all_values.append(true_val)
            true_arg_vals = true_val.arg_values
            for arg_pos in range(operation.arity):
              assert true_arg_vals[arg_pos] in all_value_dict
              true_args.append(all_value_dict[true_arg_vals[arg_pos]])
            for arg_pos in range(operation.arity):
              true_args += [ARGV_MAP[argv] for argv in true_val.arg_variables[arg_pos]]
              true_args += [-1] * (MAX_NUM_ARGVS - len(true_val.arg_variables[arg_pos]))
            true_args = np.array(true_args, dtype=np.int32)
            args = np.concatenate((args, np.expand_dims(true_args, 0)), axis=0)
            trace_in_beam = args.shape[0] - 1
          training_samples.append((args, weight_snapshot, trace_in_beam, num_values_before_op, operation))
        trace_values.append(true_val)
        trace.pop(0)
        if len(trace) == 0:
          return training_samples, all_values, stats
    if len(all_values) == cur_num_values and not use_ur and not is_stochastic:
      # no improvement
      break
  return None, all_values, stats
