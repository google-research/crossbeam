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
import timeit

from crossbeam.dsl import value as value_module


def _add_value_by_weight(values_by_weight, value):
  if value.get_weight() < len(values_by_weight):
    values_by_weight[value.get_weight()][value] = value


def _gather_values_with_weight_and_type(values_by_weight, arg_weight, arg_type,
                                        cache):
  key = (arg_weight, arg_type)
  if key in cache:
    return cache[key]
  if arg_type is None:
    values = list(values_by_weight[arg_weight])
  else:
    values = [v for v in values_by_weight[arg_weight] if v.type == arg_type]
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


def synthesize_baseline(task, domain, max_weight=10, timeout=5,
                        max_values_explored=None):
  """Synthesizes a solution using normal bottom-up enumerative search."""
  start_time = timeit.default_timer()
  end_time = start_time + timeout if timeout else None
  num_examples = task.num_examples
  stats = {'num_values_explored': 0}

  # A list of OrderedDicts mapping Value objects to themselves. The i-th
  # OrderedDict contains all Value objects of weight i.
  values_by_weight = [collections.OrderedDict()
                      for _ in range(max_weight + 1)]

  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants_extractor is None:
    constants_extractor = lambda unused_inputs_dict: constants
  for constant in constants_extractor(task):
    _add_value_by_weight(values_by_weight,
                         value_module.ConstantValue(constant,
                                                    num_examples=num_examples))

  for input_name, input_value in task.inputs_dict.items():
    _add_value_by_weight(values_by_weight,
                         value_module.InputValue(input_value, name=input_name))

  # A set storing all values found so far.
  value_set = set().union(*values_by_weight)
  typechecking_cache = {}

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
    for op in domain.operations:

      arity = op.arity
      arg_types = op.arg_types()
      if arg_types is None:
        arg_types = [None] * op.arity

      if target_weight - op.weight - arity < 0:
        continue  # Not enough weight to use this op.

      # Enumerate ways of partitioning (target_weight - self.weight) into
      # (arity) positive pieces.
      # Equivalently, partition (target_weight - op.weight - arity) into
      # (arity) nonnegative pieces.
      for arg_weights_minus_1 in generate_partitions(
          target_weight - op.weight - arity, arity):

        if (end_time is not None and timeit.default_timer() > end_time) or (
            max_values_explored is not None and stats['num_values_explored'] >= max_values_explored):
          return None, value_set, values_by_weight, stats

        arg_options_list = []
        arg_weights = [w + 1 for w in arg_weights_minus_1]
        for arg_weight, arg_type in zip(arg_weights, arg_types):
          arg_options_list.append(_gather_values_with_weight_and_type(
              values_by_weight, arg_weight, arg_type, typechecking_cache))

        for arg_list in itertools.product(*arg_options_list):
          value = op.apply(arg_list)
          stats['num_values_explored'] += 1

          if value is None or value in value_set:
            continue
          if (domain.small_value_filter and
              not all(domain.small_value_filter(v) for v in value.values)):
            continue

          if value == output_value:
            # Found solution!
            return value, value_set, values_by_weight, stats

          values_by_weight[target_weight][value] = value
          value_set.add(value)

    print('Bottom-up enumeration found {} distinct tasks of weight {}, or {} '
          'distinct tasks total, in {:.2f} seconds total'.format(
              len(values_by_weight[target_weight]), target_weight,
              len(value_set), timeit.default_timer() - start_time))

  return None, value_set, values_by_weight, stats
