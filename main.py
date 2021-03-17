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

import numpy as np

import operation as operation_module
import unique_randomizer as ur
import value as value_module


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
  for _ in range(k):
    if randomizer.exhausted():
      break
    sequence = [all_values[randomizer.sample_distribution(dist)]
                for _ in range(arity)]
    randomizer.mark_sequence_complete()
    beam.append(sequence)
  return beam


DUPLICATES = 0


def synthesize(inputs_dict, output, max_weight=10, k=10, verbose=False):
  """Synthesizes an expression that creates the output."""
  global DUPLICATES
  randomizer = ur.UniqueRandomizer()

  all_values = []
  all_values.append(value_module.ConstantValue(0, weight=1))
  all_values.append(value_module.ConstantValue(1, weight=1))

  for input_name, input_value in inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, weight=1,
                                              name=input_name))

  output_value = value_module.OutputValue(output, weight=-1)

  all_values_set = set(all_values)

  while True:
    for operation in operation_module.OPERATIONS:
      beam = random_beam_search_ur(randomizer, all_values, operation, k=k)
      # beam = random_beam_search(all_values, operation, k=k)
      for arg_list in beam:
        result = operation.apply([arg.value for arg in arg_list])
        weight = 1 + sum(arg.weight for arg in arg_list)
        if weight > max_weight:
          continue
        result_value = value_module.OperationValue(
            result, weight, operation, arg_list)
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
          return result_value.expression(), all_values


def main():
  inputs_dict = {
      'five': 5,
      'seven': 7,
  }
  output = (5, ((1, 7), 0))
  print('Inputs dict: {}'.format(inputs_dict))
  print('Target output: {}'.format(output))
  solution, all_values = synthesize(inputs_dict, output,
                                    max_weight=100, k=10, verbose=False)
  print('Synthesis success! Solution: {}'.format(solution))

  weights = [v.weight for v in all_values]
  for weight in range(max(weights) + 1):
    print('Number of values with weight {}: {}'.format(
        weight, sum(w == weight for w in weights)))

  print('Num duplicates: {}'.format(DUPLICATES))

if __name__ == '__main__':
  main()
