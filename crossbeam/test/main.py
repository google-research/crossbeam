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

"""CrossBeam synthesizer."""

import random
import timeit

import numpy as np

from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import tuple_operations
from crossbeam.dsl import value as value_module
from crossbeam.unique_randomizer import unique_randomizer as ur


def random_beam_search(all_values, operation, k):
  arity = operation.arity
  return [
      [random.choice(all_values) for _ in range(arity)]
      for _ in range(k)
  ]


def random_beam_search_ur(randomizer, all_values, operation, k):
  arity = operation.arity
  beam = []
  dist = np.exp(-np.array([v.weight for v in all_values]))

  # Let the root node know that more children are available, if it was
  # previously exhausted.
  randomizer.sample_distribution(dist)
  randomizer.clear_sequence()

  for _ in range(k):
    if randomizer.exhausted():
      break
    sequence = [all_values[randomizer.sample_distribution(dist)]
                for _ in range(arity)]
    randomizer.mark_sequence_complete()
    beam.append(sequence)
  return beam


DUPLICATES = 0


def synthesize(task, operations, constants, max_weight=10, k=10, verbose=False,
               time_limit=30):
  """Synthesizes an expression that creates the output."""
  global DUPLICATES
  start_time = timeit.default_timer()

  randomizers = [ur.UniqueRandomizer() for _ in operations]  # One per op.
  num_examples = task.num_examples

  all_values = []
  for constant in constants:
    all_values.append(value_module.ConstantValue(constant,
                                                 num_examples=num_examples))

  for input_name, input_value in task.inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, name=input_name))

  output_value = value_module.OutputValue(task.outputs)

  all_values_set = set(all_values)

  while True:
    for operation, randomizer in zip(operations, randomizers):

      if timeit.default_timer() - start_time >= time_limit:
        return None, all_values

      beam = random_beam_search_ur(randomizer, all_values, operation, k=k)
      # beam = random_beam_search(all_values, operation, k=k)
      for arg_list in beam:
        result_value = operation.apply(arg_list)
        if result_value is None or result_value.weight > max_weight:
          continue
        if result_value in all_values_set:
          # TODO: replace existing one if this way is simpler (less weight)
          DUPLICATES += 1
          continue
        all_values_set.add(result_value)
        all_values.append(result_value)

        if verbose:
          print('Found value: {}, code {}'.format(result_value.value,
                                                  result_value.expression()))

        if result_value == output_value:
          return result_value, all_values


def main():
  operations = tuple_operations.get_operations()
  min_task_weight = 3
  max_task_weight = random.randint(6, 7)
  max_search_weight = 8
  constants = [0]

  for _ in range(10):
    task = random_data.generate_random_task(
        min_weight=min_task_weight,
        max_weight=max_task_weight,
        num_examples=2,
        num_inputs=random.randint(1, 2),
        constants=constants,
        operations=operations,
        input_generator=random_data.RANDOM_INTEGER)
    if task:
      break
  if task is None:
    print('Could not create random task.')
    return

  # TODO: Should check that no input equals another input or a given constant
  # across all examples.
  # TODO: Should check that the output doesn't exactly equal an input or
  # constant across all examples.

  print(task)

  start_time = timeit.default_timer()
  solution, all_values = synthesize(
      task,
      operations=operations, constants=constants,
      max_weight=max_search_weight, k=20, verbose=False)
  if solution:
    print('Synthesis success! Solution: {}, weight: {}'.format(
        solution.expression(), solution.weight))
  else:
    print('Synthesis failed.')
  print('Elapsed time: {:.2f} sec'.format(timeit.default_timer() - start_time))

  dp_info = random_data.num_expressions_dp(
      operations=operations,
      num_inputs=task.num_inputs,
      constants=constants,
      max_weight=max_search_weight)
  weights = [v.weight for v in all_values]
  print()
  for weight in range(max(weights) + 1):
    num_with_weight = sum(w == weight for w in weights)
    total_with_weight = dp_info.num_expressions[weight]
    print('Number of values with weight {}: {} / {}'.format(
        weight, num_with_weight, total_with_weight))
    assert num_with_weight <= total_with_weight

  print('Num duplicates: {}'.format(DUPLICATES))

if __name__ == '__main__':
  main()
