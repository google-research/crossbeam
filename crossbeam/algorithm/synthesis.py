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
from crossbeam.algorithm.beam_search import beam_search, batch_beam_search
from crossbeam.dsl import value as value_module
from crossbeam.unique_randomizer import unique_randomizer as ur


def init_values(task, domain, all_values):
  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  for constant in constants_extractor(task):
    all_values.append(value_module.ConstantValue(constant,
                                                 num_examples=task.num_examples))
  for input_name, input_value in task.inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, name=input_name))
  output_value = value_module.OutputValue(task.outputs)
  return output_value


def op_in_beam_synthesize(task, domain, model, device,
                          trace=None, max_weight=10, k=2, is_training=False,
                          include_as_train=None, timeout=None, is_stochastic=False,
                          random_beam=False):
  end_time = None if timeout is None or timeout < 0 else timeit.default_timer() + timeout
  if trace is None:
    trace = []
  if include_as_train is None:
    include_as_train = lambda trace_in_beam: True

  all_values = deepcopy(domain.operations)
  max_beam_steps = max([op.arity for op in domain.operations]) + 1

  output_value = init_values(task, domain, all_values)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  if not random_beam:
    io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
  training_samples = []

  val_embed = model.val(all_values, device=device)
  while True:
    cur_num_values = len(all_values)
    num_values_before_op = len(all_values)
    if random_beam:
      val_offset = len(domain.operations)
      args = [[np.random.randint(len(domain.operations))] for _ in range(k)]
      for b in range(k):
        args[b] += [np.random.randint(val_offset, len(all_values)) for _ in range(max_beam_steps)]
    else:
      if len(all_values) > val_embed.shape[0]:
        more_val_embed = model.val(all_values[val_embed.shape[0]:], device=device)
        val_embed = torch.cat((val_embed, more_val_embed), dim=0)
      op_state = model.init(io_embed, val_embed, 'dummy')
      args, _ = beam_search(max_beam_steps, k,
                            val_embed,
                            op_state,
                            model.arg,
                            device=device,
                            is_stochastic=is_stochastic)
      args = args.data.cpu().numpy().astype(np.int32)
    if k > (len(all_values) ** max_beam_steps):
      args = args[:len(all_values) ** max_beam_steps]
    beam = [[all_values[i] for i in arg_list] for arg_list in args]
    trace_in_beam = -1
    for i, arg_list in enumerate(beam):
      operation = arg_list[0]
      arg_list = arg_list[1:operation.arity + 1]
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
    if end_time is not None and timeit.default_timer() > end_time:
      return None, None
    if is_training and len(trace):
      if include_as_train(trace_in_beam):  # construct training example
        if trace_in_beam < 0:  # true arg not found
          true_args = [all_value_dict[operation]]
          operation = trace[0].operation
          true_val = trace[0]
          if not true_val in all_value_dict:
            all_value_dict[true_val] = len(all_values)
            all_values.append(true_val)
          true_arg_vals = true_val.arg_values
          for i in range(operation.arity):
            assert true_arg_vals[i] in all_value_dict
            true_args.append(all_value_dict[true_arg_vals[i]])
          true_args += [0] * (max_beam_steps - len(true_args))
          true_args = np.array(true_args, dtype=np.int32)
          args = np.concatenate((args, np.expand_dims(true_args, 0)), axis=0)
          trace_in_beam = args.shape[0] - 1
        decision_lens = [domain.operations[i].arity + 1 for i in args[:, 0]]
        training_samples.append((args, decision_lens, trace_in_beam, num_values_before_op, operation))
      trace.pop(0)
      if len(trace) == 0:
        return training_samples, all_values
    if len(all_values) == cur_num_values:  # no improvement
      break
  return None, None


def update_masks(type_masks, operation, all_values, device, vidx_start=0):
  feasible = True
  for arg_index in range(operation.arity):
    arg_type = operation.arg_types()[arg_index]
    bool_mask = [all_values[v].type == arg_type for v in range(vidx_start, len(all_values))]
    step_type_mask = torch.BoolTensor(bool_mask).to(device)
    if len(type_masks) <= arg_index:
      type_masks.append(step_type_mask)
      if not any(step_type_mask):
        feasible = False
    else:
      type_masks[arg_index] = torch.cat([type_masks[arg_index], step_type_mask])
      if not any(type_masks[arg_index]):
        feasible = False
  return feasible


def synthesize(task, domain, model, device,
               trace=None, max_weight=15, k=2, is_training=False,
               include_as_train=None, timeout=None, is_stochastic=False,
               random_beam=False, use_ur=False, masking=True):
  stats = {'num_values_explored': 0}

  verbose = False
  end_time = None if timeout is None or timeout < 0 else timeit.default_timer() + timeout
  if trace is None:
    trace = []
  if include_as_train is None:
    include_as_train = lambda trace_in_beam: True

  all_values = []
  output_value = init_values(task, domain, all_values)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  if not random_beam:
    io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
  training_samples = []

  val_embed = model.val(all_values, device=device)
  mask_dict = {}
  for operation in domain.operations:
    type_masks = []
    if operation.arg_types() is not None and masking:
      update_masks(type_masks, operation, all_values, device)
    mask_dict[operation] = type_masks

  while True:
    cur_num_values = len(all_values)
    for operation in domain.operations:
      if end_time is not None and timeit.default_timer() > end_time:
        return None, all_values, stats
      if verbose:
        print('Operation: {}'.format(operation))
      num_values_before_op = len(all_values)
      type_masks = mask_dict[operation]

      if len(type_masks):
        if len(all_values) > type_masks[0].shape[0]:
          feasible = update_masks(type_masks, operation, all_values, device, vidx_start=type_masks[0].shape[0])
          if not feasible:
            continue

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
              if len(type_masks):
                scores = torch.where(type_masks[arg_index], scores, torch.FloatTensor([-1e10]).to(device))
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
          stats['num_values_explored'] += 1
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
            return new_value, all_values, stats

        continue

      if random_beam:
        args = [[] for _ in range(k)]
        val_offset = 0
        for b in range(k):
          args[b] += [np.random.randint(val_offset, len(all_values)) for _ in range(operation.arity)]
      else:
        if len(all_values) > val_embed.shape[0]:
          more_val_embed = model.val(all_values[val_embed.shape[0]:], device=device)
          val_embed = torch.cat((val_embed, more_val_embed), dim=0)
        op_state = model.init(io_embed, val_embed, operation)
        args, _ = beam_search(operation.arity, k,
                              val_embed,
                              op_state,
                              model.arg,
                              device=device,
                              choice_masks=type_masks,
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
          return result_value, all_values, stats
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
          training_samples.append((args, [], trace_in_beam, num_values_before_op, operation))
        trace.pop(0)
        if len(trace) == 0:
          return training_samples, all_values, stats
    if len(all_values) == cur_num_values:  # no improvement
      break
  return None, all_values, stats


def get_or_add_value(val, all_values, all_value_dict):
  if val in all_value_dict:
    return all_value_dict[val]
  idx = len(all_values)
  all_value_dict[val] = idx
  all_values.append(val)
  return idx


def batch_synthesize(tasks, domain, model, device, traces=None, max_weight=10, k=2, is_training=False,
                     include_as_train=None, timeout=None, is_stochastic=False, random_beam=False, masking=True):
  end_time = None if timeout is None or timeout < 0 else timeit.default_timer() + timeout
  if traces is None:
    trace = [[] for _ in range(len(tasks))]
  if include_as_train is None:
    include_as_train = lambda trace_in_beam: True
  all_values = []
  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  all_values = []
  all_value_dict = {}

  output_values = []
  value_indices = []
  task_done = [False] * len(tasks)
  for task in tasks:
    num_examples = task.num_examples
    indices = []
    for constant in constants_extractor(task):
      idx = get_or_add_value(value_module.ConstantValue(constant,
                                                        num_examples=num_examples),
                             all_values, all_value_dict)
      indices.append(idx)
    for input_name, input_value in task.inputs_dict.items():
      idx = get_or_add_value(value_module.InputValue(input_value, name=input_name),
                             all_values, all_value_dict)
      indices.append(idx)
    value_indices.append(indices)
    output_values.append(value_module.OutputValue(task.outputs))
  if not random_beam:
    io_embed, io_scatter = model.io([task.inputs_dict for task in tasks],
                                    [task.outputs for task in tasks],
                                    device=device,
                                    needs_scatter_idx=True)

    val_embed = model.val(all_values, device=device)
  mask_dict = {}
  for operation in domain.operations:
    type_masks = []
    if operation.arg_types() is not None and masking:
      update_masks(type_masks, operation, all_values, device)
    mask_dict[operation] = type_masks

  max_beam_steps = max([operation.arity for operation in domain.operations])
  training_samples = []
  while not all(task_done):
    for operation in domain.operations:
      if all(task_done):
        break
      type_masks = mask_dict[operation]
      if len(type_masks) and len(all_values) > type_masks[0].shape[0]:
        if not update_masks(type_masks, operation, all_values, device, vidx_start=type_masks[0].shape[0]):
          continue

      cur_val_indices = [torch.LongTensor(vid).to(device) for vid in value_indices]
      active_tasks = []
      for t_idx, task in enumerate(tasks):
        if task_done[t_idx]:
          continue
        vid = cur_val_indices[t_idx]
        task_feasible = True
        for step in range(operation.arity):
          if torch.max(type_masks[step][vid]) + 1e-8 < 1.0:
            task_feasible = False
            break
        if task_feasible:
          active_tasks.append(t_idx)
      if len(active_tasks) == 0:
        continue
      if not random_beam:
        if len(all_values) > val_embed.shape[0]:
          more_val_embed = model.val(all_values[val_embed.shape[0]:], device=device)
          val_embed = torch.cat((val_embed, more_val_embed), dim=0)        
        op_states = model.batch_init(io_embed, io_scatter, val_embed, cur_val_indices, operation,
                                     sample_indices=active_tasks)
        batch_beam, _ = batch_beam_search(operation.arity, k,
                                          val_embed,
                                          [cur_val_indices[v] for v in active_tasks],
                                          op_states,
                                          model.arg,
                                          device=device,
                                          choice_masks=type_masks,
                                          is_stochastic=is_stochastic)
      for local_tid, t_idx in enumerate(active_tasks):
        trace = traces[t_idx]
        output_value = output_values[t_idx]
        if random_beam:
          args = [[] for _ in range(k)]
          for b in range(k):
            args[b] += [np.random.choice(value_indices[t_idx]) for _ in range(operation.arity)]
          args = np.array(args, dtype=np.int32)
        else:
          args = batch_beam[local_tid]
        beam = [[all_values[i] for i in arg_list] for arg_list in args]
        trace_in_beam = -1
        for bid, arg_list in enumerate(beam):
          result_value = operation.apply(arg_list)
          if result_value is None or result_value.weight > max_weight:
            continue
          if (domain.small_value_filter and
              not all(domain.small_value_filter(v) for v in result_value.values)):
            continue
          if result_value in all_value_dict:
            continue
          vid = get_or_add_value(result_value, all_values, all_value_dict)
          value_indices[t_idx].append(vid)
          if result_value == output_value and not is_training:
            task_done[t_idx] = True
          else:
            if len(trace) and result_value == trace[0] and trace_in_beam < 0:
              trace_in_beam = bid
        if end_time is not None and timeit.default_timer() > end_time:
          return None, None
        if is_training and len(trace) and trace[0].operation == operation:
          if include_as_train(trace_in_beam):
            if trace_in_beam < 0:  # true arg not found
              true_args = []
              true_val = trace[0]
              vid = get_or_add_value(true_val, all_values, all_value_dict)
              value_indices[t_idx].append(vid)
              true_arg_vals = true_val.arg_values
              for i in range(operation.arity):
                assert true_arg_vals[i] in all_value_dict
                true_args.append(all_value_dict[true_arg_vals[i]])
              true_args = np.array(true_args, dtype=np.int32)
              args = np.concatenate((args, np.expand_dims(true_args, 0)), axis=0)
              trace_in_beam = args.shape[0] - 1
            args = np.pad(args, [(0, 0), (0, max_beam_steps - operation.arity)])
            training_samples.append((t_idx, args, trace_in_beam, value_indices[t_idx][:], operation))
          trace.pop(0)
          if len(trace) == 0:
            task_done[t_idx] = True
  return training_samples, all_values
