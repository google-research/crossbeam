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

"""Generates random data."""

import bisect
import collections
import functools
import itertools
import operator
import random
from typing import List

from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module

# answer = the number of expressions with weight <= the max weight.
# num_expressions[i] = number of expressions with weight exactly i.
# cumulative_expressions[i] = number of expressions with weight <= i.
# op_table[i][j] = number of expressions with weight exactly i, using any
#     operation index 0 through j (inclusive) at the root.
# partition_table[i][j][k] = number of expressions with weight exactly i, using
#     operation index j at the root, using any partition index 0 through k.
DpInfo = collections.namedtuple(
    'DpInfo', ['answer', 'num_expressions', 'cumulative_expressions',
               'op_table', 'partition_table'])


@functools.lru_cache(maxsize=None)
def generate_partitions(num_elements: int, num_parts: int) -> List[List[int]]:
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


def num_expressions_dp(operations, num_inputs, constants, max_weight):
  """Computes the number of expressions up to a max weight."""
  # num_expressions[i] is the number of expressions of weight i.
  num_expressions = [0] * (1 + max_weight)
  # The only expressions of weight 1 are constants and inputs.
  num_expressions[1] += len(constants) + num_inputs

  # op_table[i][j] is the number of expressions of weight i, using any operation
  # index 0 through j (inclusive) at the root.
  op_table = [[0] * len(operations) for _ in range(1 + max_weight)]

  # partition_table[i][j][k] = number of expressions with weight exactly i,
  # using operation index j at the root, using any partition index 0 through k.
  partition_table = [[None] * len(operations) for _ in range(1 + max_weight)]

  for total_weight in range(1, max_weight + 1):
    for op_index, op in enumerate(operations):
      op_weight = op.weight
      op_arity = op.arity
      if total_weight - op_weight < op_arity:
        continue

      # Partition `total_weight - op_weight` into `op_arity` positive pieces.
      # Equivalently, partition `total_weight - op_weight - op_arity` into
      # `op_arity` nonnegative pieces.
      partitions = generate_partitions(total_weight - op_weight - op_arity,
                                       op_arity)
      partition_table[total_weight][op_index] = [0] * len(partitions)

      for partition_index, partition in enumerate(partitions):
        arg_weights = [part + 1 for part in partition]
        num_for_op = functools.reduce(
            operator.mul, (num_expressions[w] for w in arg_weights))
        partition_table[total_weight][op_index][partition_index] = num_for_op
        op_table[total_weight][op_index] += num_for_op
        num_expressions[total_weight] += num_for_op

      partition_table[total_weight][op_index] = list(itertools.accumulate(
          partition_table[total_weight][op_index]))

    op_table[total_weight] = list(itertools.accumulate(op_table[total_weight]))

  cumulative_expressions = list(itertools.accumulate(num_expressions))
  return DpInfo(answer=cumulative_expressions[max_weight],
                num_expressions=num_expressions,
                cumulative_expressions=cumulative_expressions,
                op_table=op_table,
                partition_table=partition_table)


def generate_value_with_index(inputs_dict, constants, num_examples,
                              operations, dp_info, value_index):
  """Generates a value with the given index."""
  num_expressions = dp_info.num_expressions
  cumulative_expressions = dp_info.cumulative_expressions
  op_table = dp_info.op_table
  partition_table = dp_info.partition_table

  weight = bisect.bisect(cumulative_expressions, value_index)

  if weight == 1:  # Base case - not using an operation.
    if value_index < len(constants):
      return value_module.ConstantValue(constants[value_index],
                                        num_examples=num_examples)
    else:
      value_index -= len(constants)
      name = list(inputs_dict)[value_index]
      return value_module.InputValue(inputs_dict[name], name)

  value_index -= cumulative_expressions[weight - 1]
  op_index = bisect.bisect(op_table[weight], value_index)
  op = operations[op_index]

  value_index -= op_table[weight][op_index - 1] if op_index > 0 else 0
  partition_index = bisect.bisect(partition_table[weight][op_index],
                                  value_index)
  partitions = generate_partitions(weight - op.weight - op.arity, op.arity)
  partition = partitions[partition_index]

  arg_values = []
  for arg_index in range(op.arity):
    arg_weight = partition[arg_index] + 1
    subvalue_index = (cumulative_expressions[arg_weight - 1]
                      + value_index % num_expressions[arg_weight])
    arg_value = generate_value_with_index(
        inputs_dict, constants, num_examples, operations, dp_info,
        value_index=subvalue_index)
    if arg_value is None:
      return None  # A subexpression failed to execute.
    arg_values.append(arg_value)
    value_index //= num_expressions[arg_weight]

  return op.apply(arg_values)  # This may be None!


RANDOM_INTEGER = functools.partial(random.randint, a=0, b=9)


def generate_random_task(domain, min_weight, max_weight, num_examples,
                         num_inputs, dp_info=None, random_seed=None):
  """Generate a random Examples object."""
  if random_seed is not None:
    random.seed(random_seed)

  inputs_dict = {
      'in{}'.format(input_index + 1):
          [domain.input_generator() for _ in range(num_examples)]
      for input_index in range(num_inputs)
  }

  constants = domain.constants
  constants_extractor = domain.constants_extractor
  assert (constants is None) != (constants_extractor is None), (
      'expected exactly one of constants or constants_extractor')
  if constants is None:
    constants = constants_extractor(inputs_dict)

  if dp_info is None:
    dp_info = num_expressions_dp(domain.operations, num_inputs, constants,
                                 max_weight)

  min_index = dp_info.cumulative_expressions[min_weight - 1]
  max_index = dp_info.cumulative_expressions[max_weight] - 1
  if min_index >= max_index:
    return None
  value_index = random.randint(min_index, max_index)
  solution_value = generate_value_with_index(
      inputs_dict, constants, num_examples, domain.operations, dp_info,
      value_index)
  if solution_value is None:
    return None
  return task_module.Task(inputs_dict, solution_value.values,
                          solution=solution_value)


def _duplicate_check_dfs(node, ancestors):
  """Returns whether any node equals any of its descendants."""
  if node in ancestors:
    return True
  if isinstance(node, value_module.OperationValue):
    ancestors.append(node)
    for arg_node in node.arg_values:
      if _duplicate_check_dfs(arg_node, ancestors):
        return True
    del ancestors[-1]
  return False


def generate_good_random_task(**kwargs):
  """Generates a task that passes simple quality checks."""
  while True:
    task = generate_random_task(**kwargs)
    if task is None:
      continue

    # For any input or output, it cannot be identical to another input or a
    # constant.
    inputs = list(task.inputs_dict.values())

    domain = kwargs['domain']
    constants = domain.constants
    constants_extractor = domain.constants_extractor
    assert (constants is None) != (constants_extractor is None), (
        'expected exactly one of constants or constants_extractor')
    if constants is None:
      constants = constants_extractor(task.inputs_dict)
    good = True
    for to_check, other in itertools.product(inputs + task.outputs,
                                             inputs + constants):
      if to_check is not other and to_check == other:
        good = False
        break
    if not good:
      continue

    # The output can't be a constant across all examples.
    num_examples = kwargs['num_examples']
    if (num_examples > 1 and
        all(x == y for x, y in itertools.combinations(task.outputs, 2))):
      continue

    # A node in the AST can't have descendants that are equal to it, or else the
    # solution isn't minimal.
    if _duplicate_check_dfs(task.solution, []):
      continue

    # The task has passed all checks.
    return task
