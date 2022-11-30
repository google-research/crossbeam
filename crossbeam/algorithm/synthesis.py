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

import copy
import timeit

import numpy as np
import torch

from crossbeam.algorithm.beam_search import beam_search
from crossbeam.dsl import value as value_module
from crossbeam.unique_randomizer import unique_randomizer as ur
from crossbeam.property_signatures import property_signatures
from crossbeam.algorithm.variables import MAX_NUM_FREE_VARS, MAX_NUM_ARGVS, ALL_BOUND_VARS, ALL_FREE_VARS, ARGV_MAP


def init_values(task, domain, all_values):
  """Puts initial values into `all_values` and returns the output value."""
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
  """Do type checking for new values starting at index `vidx_start`."""
  feasible = True  # Do we have a value of the right type, for every argument?
  for arg_index in range(operation.arity):
    arg_type = operation.arg_types()[arg_index]
    bool_mask = [arg_type is None or all_values[v].type == arg_type
                 for v in range(vidx_start, len(all_values))]
    cur_feasible = any(bool_mask)
    step_type_mask = torch.BoolTensor(bool_mask).to(device)
    if len(type_masks) <= arg_index:
      type_masks.append([cur_feasible, step_type_mask])
    else:
      type_masks[arg_index][1] = torch.cat([type_masks[arg_index][1],
                                            step_type_mask])
      cur_feasible = cur_feasible or type_masks[arg_index][0]
      type_masks[arg_index][0] = cur_feasible
    feasible = feasible and cur_feasible
  return feasible


def update_with_better_value(result_value, all_value_dict, all_values, verbose):
  """Update a value with a better (less weight) way of constructing it."""
  old_value = all_values[all_value_dict[result_value]]
  if result_value.get_weight() >= old_value.get_weight():
    # New value is not better than the old one. Nothing to do here.
    return

  assert isinstance(old_value, value_module.OperationValue)
  if verbose:
    print('duplicate value found. was: {}, {}, weight {}'.format(
        old_value, old_value.expression(), old_value.get_weight()))
  old_value.operation = result_value.operation
  old_value.arg_values = list(result_value.arg_values)
  old_value.arg_variables = list(result_value.arg_variables)
  old_value.contains_lambda = result_value.contains_lambda
  old_value._repr_cache = None  # pylint: disable=protected-access
  if verbose:
    print('  updated to: {}, {}, weight {}'.format(
        old_value, old_value.expression(), old_value.get_weight()))


def copy_operation_value(operation, value, all_values, all_value_dict,
                         trace_values):
  """Copy an OperationValue to avoid modifying the original."""
  assert isinstance(value, value_module.OperationValue)
  arg_values = []
  for v in value.arg_values:
    # TODO(kshi): line below is only needed because value's repr format changed
    # between dataset generation and training.
    v._repr_cache = None  # pylint: disable=protected-access
    if v in all_value_dict:
      arg_values.append(all_values[all_value_dict[v]])
    else:
      arg_values.append(trace_values[v])
  if not value.values:
    return operation.apply(arg_values, value.arg_variables,
                           value.free_variables)
  else:
    return value_module.OperationValue(
        value.values, value.operation, arg_values,
        arg_variables=copy.deepcopy(value.arg_variables),
        free_variables=copy.deepcopy(value.free_variables))


def decode_args(operation, args, all_values):
  """Parse out variables from the argument list."""
  arg_list = args[:operation.arity]
  arg_list = [all_values[x] for x in arg_list]
  offset = operation.arity
  free_vars = set([v for v in arg_list
                   if isinstance(v, value_module.FreeVariable)])
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


def update_stats_value_explored(stats, value):
  stats['num_values_explored'] += 1
  if value is None:
    stats['num_explored_none'] += 1
  elif value.num_free_variables:
    stats['num_explored_lambda'] += 1
  else:
    stats['num_explored_concrete'] += 1


def update_stats_value_kept(stats, value):
  # Not including trace elements manually added during training.
  stats['num_values_kept'] += 1
  if value.num_free_variables:
    stats['num_kept_lambda'] += 1
  else:
    stats['num_kept_concrete'] += 1


def update_stats_with_percents(stats):
  stats.update({
      'explored_percent_none':
          stats['num_explored_none'] * 100 / stats['num_values_explored']
          if stats['num_values_explored'] else -1,
      'explored_percent_concrete':
          stats['num_explored_concrete'] * 100 / stats['num_values_explored']
          if stats['num_values_explored'] else -1,
      'explored_percent_lambda':
          stats['num_explored_lambda'] * 100 / stats['num_values_explored']
          if stats['num_values_explored'] else -1,
      'kept_percent_concrete':
          stats['num_kept_concrete'] * 100 / stats['num_values_kept']
          if stats['num_values_kept'] else -1,
      'kept_percent_lambda':
          stats['num_kept_lambda'] * 100 / stats['num_values_kept']
          if stats['num_values_kept'] else -1,
  })


def synthesize(task, domain, model, device,
               trace=None, max_weight=15, k=2, is_training=False,
               timeout=None, max_values_explored=None, is_stochastic=False,
               random_beam=False, use_ur=False, masking=True,
               static_weight=False):
  """Perform CrossBeam synthesis."""

  stats = {
      'num_examples': task.num_examples,
      'num_inputs': task.num_inputs,
      'num_values_explored': 0,
      'num_explored_none': 0,
      'num_explored_concrete': 0,
      'num_explored_lambda': 0,
      'num_values_kept': 0,
      'num_kept_concrete': 0,
      'num_kept_lambda': 0,
  }
  verbose = False
  end_time = (None if timeout is None or timeout < 0
              else timeit.default_timer() + timeout)

  # We can use at most 1 of:
  #   * random_beam (sampling during search)
  #   * is_stochastic (sampling during evaluation)
  #   * use_ur (UniqueRandomizer during evaluation)
  assert int(random_beam) + int(is_stochastic) + int(use_ur) <= 1

  # Initialize collections that store all values found during search.
  all_values = []
  output_value = init_values(task, domain, all_values)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  # For collecting training data from ground-truth traces.
  if trace is None:
    trace = []
  trace_values = {}
  training_samples = []

  # For type checking if the operations provide type signatures.
  mask_dict = {}
  for operation in domain.operations:
    type_masks = []
    if operation.arg_types() is not None and masking:
      update_masks(type_masks, operation, all_values, device)
    mask_dict[operation] = type_masks

  # Initialize the model with the I/O example and initial values.
  all_signatures = []
  if not random_beam:
    io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
    val_base_embed, all_signatures = model.val(
        all_values, device=device, output_values=output_value,
        need_signatures=True)

  # Main synthesis loop.
  last_num_values_before_operation_loop = -1
  while True:

    # If using plain beam search, and we didn't find any new values for an
    # entire loop over all operations, then we're stuck and won't make any
    # further progress.
    if (not use_ur and not is_stochastic and
        len(all_values) == last_num_values_before_operation_loop):
      break
    last_num_values_before_operation_loop = len(all_values)

    # Loop over all operations.
    for operation in domain.operations:

      # Check for exceeding computation limit.
      if (end_time is not None and timeit.default_timer() > end_time) or (
          max_values_explored is not None
          and stats['num_values_explored'] >= max_values_explored):
        return None, (all_values, all_signatures), stats

      if verbose:
        print('Operation: {}'.format(operation))

      # Type checking.
      type_masks = mask_dict[operation]
      if type_masks and len(all_values) > type_masks[0][1].shape[0]:
        feasible = update_masks(type_masks, operation, all_values, device,
                                vidx_start=type_masks[0][1].shape[0])
        if not feasible:
          # We don't have appropriately-typed values needed for the operation.
          continue

      # Info about search context before we do beam search, used to create
      # training data.
      weight_snapshot = [v.get_weight() for v in all_values]
      num_values_before_op = len(all_values)
      collect_training_data_for_this_operation = (
          is_training and trace and trace[0].operation == operation)

      # Get argument lists via beam search, random sampling, or
      # UniqueRandomizer, by using a generator.
      if random_beam:
        # TODO(hadai): enable random beam during training
        raise NotImplementedError()
      elif use_ur:
        frozen_all_values = list(all_values)  # Don't use new values.
        def arg_list_generator_fn(operation, weight_snapshot):
          nonlocal val_base_embed, all_signatures
          # Run the model on values it hasn't seen before.
          if len(all_values) > val_base_embed.shape[0]:
            more_val_embed, more_signatures = model.val(
                all_values[val_base_embed.shape[0]:],
                device=device, output_values=output_value, need_signatures=True)
            all_signatures += more_signatures
            val_base_embed = torch.cat((val_base_embed, more_val_embed), dim=0)
          value_embed = model.encode_weight(val_base_embed, weight_snapshot)
          op_state = model.init(io_embed, value_embed, operation)
          # Draw samples with UR incrementally, until we find enough new values
          # (which is checked outside this generator).
          randomizer = ur.UniqueRandomizer()
          while not randomizer.exhausted():
            beam = beam_search(operation.arity, 1,
                               frozen_all_values,  # pylint: disable=cell-var-from-loop
                               value_embed,
                               model.special_var_embed,
                               op_state,
                               model.arg,
                               device=device,
                               choice_masks=type_masks,
                               is_stochastic=is_stochastic,
                               randomizer=randomizer)
            randomizer.mark_sequence_complete()
            beam = beam.data.cpu().numpy().astype(np.int32)
            assert len(beam) == 1
            yield beam[0]
      else:
        # Normal beam search or sampling.
        def arg_list_generator_fn(operation, weight_snapshot):
          nonlocal val_base_embed, all_signatures
          # Run the model on values it hasn't seen before.
          if len(all_values) > val_base_embed.shape[0]:
            more_val_embed, more_signatures = model.val(
                all_values[val_base_embed.shape[0]:],
                device=device, output_values=output_value, need_signatures=True)
            all_signatures += more_signatures
            val_base_embed = torch.cat((val_base_embed, more_val_embed), dim=0)
          value_embed = model.encode_weight(val_base_embed, weight_snapshot)
          op_state = model.init(io_embed, value_embed, operation)
          # Perform beam search.
          beam = beam_search(operation.arity, k,
                             all_values,
                             value_embed,
                             model.special_var_embed,
                             op_state,
                             model.arg,
                             device=device,
                             choice_masks=type_masks,
                             is_stochastic=is_stochastic)
          for args_and_vars in beam.data.cpu().numpy().astype(np.int32):
            yield args_and_vars

      arg_list_generator = arg_list_generator_fn(operation, weight_snapshot)

      # Get argument lists and process them to create new values.
      trace_index_in_beam = -1
      beam_index = -1

      beam = []

      while True:
        beam_index += 1

        # When using UniqueRandomizer, stop once we have k new values, or we
        # tried too many times.
        if use_ur and (len(all_values) - len(frozen_all_values) >= k
                       or beam_index >= 10*k):
          break

        # Get a new argument list.
        args_and_vars = next(arg_list_generator, None)
        if args_and_vars is None:  # Generator is exhausted.
          break
        beam.append(args_and_vars)

        # Create the new value.
        arg_list, arg_vars, free_vars = decode_args(operation, args_and_vars,
                                                    all_values)
        result_value = operation.apply(arg_list, arg_vars, free_vars)
        update_stats_value_explored(stats, result_value)

        # Check various reasons to throw away this new value.
        if result_value is None or result_value.get_weight() > max_weight:
          continue
        if (domain.small_value_filter and
            not all(domain.small_value_filter(v) for v in result_value.values)):
          continue
        if result_value in all_value_dict:
          if not static_weight:
            update_with_better_value(result_value, all_value_dict, all_values,
                                     verbose)
          continue
        if not property_signatures.is_value_valid(result_value):
          continue

        # The new value is good, save it.
        all_value_dict[result_value] = len(all_values)
        all_values.append(result_value)
        update_stats_value_kept(stats, result_value)

        # Check if we found a solution.
        if result_value == output_value and not is_training:
          return result_value, (all_values, all_signatures), stats

        # Search for the next trace element.
        # TODO(hadai): allow multi-choice when options in trace have the same
        # priority. one easy fix would to include this into trace_generation
        # stage (add stochasticity)
        if (collect_training_data_for_this_operation and
            trace_index_in_beam < 0 and result_value == trace[0]):
          trace_index_in_beam = beam_index
      # End of loop over argument lists.

      # When applicable (if training and the next trace element uses this
      # operation), collect training data to increase the probability of this
      # trace element.
      if collect_training_data_for_this_operation:
        beam = np.array(beam)
        true_val = copy_operation_value(operation, trace[0], all_values,
                                        all_value_dict, trace_values)
        # If the trace element was not present in the predicted beam, add it.
        if trace_index_in_beam < 0:
          true_args = []
          if true_val not in all_value_dict:
            all_value_dict[true_val] = len(all_values)
            all_values.append(true_val)
          true_arg_vals = true_val.arg_values
          for arg_pos in range(operation.arity):
            assert true_arg_vals[arg_pos] in all_value_dict
            true_args.append(all_value_dict[true_arg_vals[arg_pos]])
          for arg_pos in range(operation.arity):
            true_args += [ARGV_MAP[argv]
                          for argv in true_val.arg_variables[arg_pos]]
            true_args += [-1] * (MAX_NUM_ARGVS
                                 - len(true_val.arg_variables[arg_pos]))
          true_args = np.array(true_args, dtype=np.int32)
          beam = np.concatenate((beam, np.expand_dims(true_args, 0)), axis=0)
          trace_index_in_beam = beam.shape[0] - 1

        # Save one element of training data.
        training_samples.append((beam, weight_snapshot, trace_index_in_beam,
                                 num_values_before_op, operation))
        trace_values[trace[0]] = true_val
        trace.pop(0)

        if not trace:
          # We've processed all trace elements, so we can return all the
          # collected training data.
          return training_samples, (all_values, all_signatures), stats

  return None, (all_values, all_signatures), stats
