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

"""A baseline bottom-up enumerative synthesizer."""

import collections
import functools
import itertools
import random
import timeit

from crossbeam.algorithm import variables
from crossbeam.dsl import value as value_module
from crossbeam.property_signatures import property_signatures


def _add_value_by_weight(values_by_weight, value):
  if value.get_weight() < len(values_by_weight):
    values_by_weight[value.get_weight()][value] = value


def _gather_values(values_by_weight, arg_weight, arg_type, max_num_free_vars,
                   cache):
  """Gathers values meeting certain criteria."""
  key = (arg_weight, arg_type, max_num_free_vars)
  if key in cache:
    return cache[key]
  if arg_type is None:
    values = [v for v in values_by_weight[arg_weight]
              if v.num_free_variables <= max_num_free_vars]
  else:
    values = [v for v in values_by_weight[arg_weight]
              if (v.type == arg_type and
                  v.num_free_variables <= max_num_free_vars)]
  cache[key] = values
  return values


@functools.lru_cache(maxsize=None)
def generate_partitions(num_elements, num_parts):
  """Generates partitions of num_elements into num_parts nonnegative parts.

  Args:
    num_elements: The number of things to permute (a nonnegative integer).
    num_parts: The number of groups to partition into (a positive integer).

  Returns:
    All possible lists of length num_parts, such that the list's elements are
    all nonnegative integers summing to num_elements.

  Raises:
    ValueError: If num_elements is negative, or num_parts is not positive.
  """
  if num_elements < 0:
    raise ValueError('In generate_partitions(), num_elements must be '
                     'nonnegative.')
  if num_parts <= 0:
    raise ValueError('In generate_partitions(), num_parts must be positive.')

  # A list [0, 1, ..., num_elements].
  choices = range(num_elements + 1)

  results = []
  # Choose (num_parts - 1) dividers among the choices, to get num_parts parts.
  for dividers in itertools.combinations_with_replacement(choices,
                                                          num_parts - 1):
    # Add dividers at the first and last choice.
    dividers = [0] + list(dividers) + [num_elements]
    # Pairwise difference between dividers gives the partition.
    results.append([next_divider - divider
                    for divider, next_divider in zip(dividers, dividers[1:])])

  return results


@functools.lru_cache(maxsize=None)
def available_variables(num_free_vars, num_bound_vars):
  return (variables.first_free_vars(num_free_vars) +
          variables.first_bound_vars(num_bound_vars))


@functools.lru_cache(maxsize=None)
def arg_vars_options(num_arg_vars, num_free_vars, num_bound_vars):
  return list(itertools.permutations(
      available_variables(num_free_vars, num_bound_vars), num_arg_vars))


def synthesize_baseline(task, domain, max_weight=10, timeout=5,
                        max_values_explored=None,
                        skip_probability=0, lambda_skip_probability=0):
  """Synthesizes a solution using normal bottom-up enumerative search."""
  print('synthesize_baseline for task: {}'.format(task))
  start_time = timeit.default_timer()
  end_time = start_time + timeout if timeout else None
  stats = {'num_values_explored': 0}

  # A list of OrderedDicts mapping Value objects to themselves. The i-th
  # OrderedDict contains all Value objects of weight i.
  values_by_weight = [collections.OrderedDict()
                      for _ in range(max_weight + 1)]

  # Add inputs before constants. If an input is equal to a constant, use the
  # input instead.
  for input_name, input_value in task.inputs_dict.items():
    _add_value_by_weight(values_by_weight,
                         value_module.InputVariable(input_value,
                                                    name=input_name))

  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  for constant in constants_extractor(task):
    _add_value_by_weight(values_by_weight, value_module.ConstantValue(constant))

  for v in variables.first_free_vars(variables.MAX_NUM_FREE_VARS):
    _add_value_by_weight(values_by_weight, v)

  # A set storing all values found so far.
  value_set = set().union(*values_by_weight)
  gather_values_cache = {}

  output_value = value_module.OutputValue(task.outputs)
  if output_value in value_set:
    # Found solution!
    match = None
    for candidate in value_set:
      if output_value == candidate:
        match = candidate
        break
    assert match is not None
    return match, value_set, values_by_weight, stats

  for target_weight in range(2, max_weight + 1):
    for num_free_vars, op in itertools.product(
        range(0, variables.MAX_NUM_FREE_VARS + 1), domain.operations):
      if target_weight >= max_weight and num_free_vars > 0:
        break

      arity = op.arity
      arg_types = op.arg_types()
      if arg_types is None:
        arg_types = [None] * op.arity
      free_vars = variables.first_free_vars(num_free_vars)
      free_vars_names = set(v.name for v in free_vars)

      remaining_weight = target_weight - op.weight - num_free_vars
      if remaining_weight - arity < 0:
        continue  # Not enough weight to use this op.

      # Enumerate ways of partitioning `remaining_weight` into `arity` positive
      # pieces.
      # Equivalently, partition `remaining_weight - arity` into `arity`
      # nonnegative pieces.
      for arg_weights_minus_1 in generate_partitions(remaining_weight - arity,
                                                     arity):

        if (end_time is not None and timeit.default_timer() > end_time) or (
            max_values_explored is not None and
            stats['num_values_explored'] >= max_values_explored):
          return None, value_set, values_by_weight, stats

        arg_options_list = []
        arg_weights = [w + 1 for w in arg_weights_minus_1]
        for arg_weight, arg_type, num_bound_variables in zip(
            arg_weights, arg_types, op.num_bound_variables):
          arg_options_list.append(_gather_values(
              values_by_weight, arg_weight, arg_type,
              num_free_vars + num_bound_variables, gather_values_cache))

        for arg_list in itertools.product(*arg_options_list):

          arg_vars_options_list = []
          for arg_index, arg in enumerate(arg_list):
            num_arg_vars = (0 if isinstance(arg, value_module.FreeVariable)
                            else arg.num_free_variables)
            arg_vars_options_list.append(
                arg_vars_options(num_arg_vars, num_free_vars,
                                 op.num_bound_variables[arg_index]))
          for arg_vars in itertools.product(*arg_vars_options_list):
            if (num_free_vars == 0 and skip_probability > 0
                and random.random() < skip_probability):
              continue
            if (num_free_vars > 0 and lambda_skip_probability > 0
                and random.random() < lambda_skip_probability):
              continue

            found_free_vars = set(v.name for v in sum(arg_vars, tuple(arg_list))
                                  if isinstance(v, value_module.FreeVariable))
            if found_free_vars != free_vars_names:
              continue

            value = op.apply(arg_list, arg_vars, free_vars)
            stats['num_values_explored'] += 1

            if value is None:
              continue
            if value.num_free_variables == 0:
              if value in value_set:
                continue
              if (domain.small_value_filter and
                  not all(domain.small_value_filter(v) for v in value.values)):
                continue
              if value == output_value:
                # Found solution!
                return value, value_set, values_by_weight, stats
            else:
              io_pairs_per_example = property_signatures.run_lambda(value)
              if not io_pairs_per_example:
                # The lambda never ran successfully, so let's skip it.
                continue

            values_by_weight[target_weight][value] = value
            value_set.add(value)
            assert value.get_weight() == target_weight

    print('Bottom-up enumeration found {} distinct tasks of weight {}, or {} '
          'distinct tasks total, in {:.2f} seconds total'.format(
              len(values_by_weight[target_weight]), target_weight,
              len(value_set), timeit.default_timer() - start_time))

  return None, value_set, values_by_weight, stats
