"""Generates held-out test cases.

This might fail to generate cases for some tasks! In that case, manually search
for `null` in the JSON and replace them with handwritten held-out test cases.
"""

import json
import os
import random
import re
from typing import Any

from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.dsl import deepcoder_utils


OUTPUT_FILENAME = 'crossbeam/data/deepcoder/held_out_test_cases.json'
NUM_CASES = 3


def json_to_str(json_data: Any, indent: int, limit: int, **kwargs) -> str:
  """Turns JSON into human-readable string with a custom max indent level."""
  json_str = json.dumps(json_data, indent=indent, **kwargs)
  # Match lines that are indented by more than the limit.
  json_str = re.sub('\n( {%s}){%d,}' % (indent, limit + 1), ' ', json_str)
  # Match lines that are indented by exactly the limit, but close a collection.
  json_str = re.sub('\n( {%s}){%d}(?=}|])' % (indent, limit), ' ', json_str)
  return json_str


def random_like(examples, num_tries):
  """Returns a random int or list[int] like the examples."""
  if isinstance(examples[0], int):
    example_min = min(examples)
    example_max = max(examples)
    if num_tries > 1000:
      example_max += 10
    if num_tries > 10000:
      example_min -= 10
    return random.randint(example_min, example_max)
  elif isinstance(examples[0], list):
    assert isinstance(examples[0][0], int)
    element_min = min(min(e) for e in examples)
    element_max = max(max(e) for e in examples)
    len_min = min(len(e) for e in examples)
    len_max = max(len(e) for e in examples)
    if num_tries > 1000:
      element_max += 10
    if num_tries > 10000:
      element_min -= 10
      len_max += 5

    # Enable generating lists with smaller range.
    bounds = [random.randint(element_min, element_max) for _ in range(3)]
    min_bound = min(bounds)
    max_bound = max(bounds)

    return [random.randint(min_bound, max_bound)
            for _ in range(random.randint(len_min, len_max))]
  else:
    raise ValueError(f'Unhandled examples: {examples}')


def generate(task):
  """Generates held-out test cases for the task."""
  input_names = list(task.inputs_dict)
  num_existing_examples = len(task.outputs)
  existing_input_tuples = [
      tuple(task.inputs_dict[name][i] for name in input_names)
      for i in range(num_existing_examples)
  ]
  existing_outputs = list(task.outputs)

  new_inputs_dict = {name: [] for name in input_names}
  new_outputs = []
  num_tries = 0

  while len(new_outputs) < NUM_CASES:
    num_tries += 1
    if num_tries > 300000:
      dummy_inputs = {name: [None] * NUM_CASES for name in input_names}
      dummy_outputs = [None] * NUM_CASES
      return dummy_inputs, dummy_outputs

    inputs_dict = {name: [random_like(task.inputs_dict[name], num_tries)]
                   for name in input_names}
    inputs_tuple = tuple(inputs_dict[name][0] for name in input_names)
    if inputs_tuple in existing_input_tuples:
      continue

    if (task.name == 'map:clip' and
        (inputs_dict['a'][0] >= inputs_dict['b'][0] or
         inputs_dict['a'][0] <= min(inputs_dict['x'][0]) or
         inputs_dict['b'][0] >= max(inputs_dict['x'][0]))):
      continue
    if (task.name == 'map:replace' and
        inputs_dict['f'][0] not in inputs_dict['x'][0]):
      continue
    if task.name == 'map:median' and len(inputs_dict['x'][0]) % 2 == 0:
      continue

    try:
      output = deepcoder_utils.run_program(task.solution, inputs_dict)[0]
    except Exception:  # pylint: disable=broad-except
      output = None
    if output is None or existing_outputs.count(output) > 1:
      continue
    if existing_outputs.count(output) > 0 and num_tries < 10000:
      continue
    if isinstance(output, int) and abs(output) > 300:
      continue
    if (isinstance(output, list) and output
        and max(abs(e) for e in output) > 300):
      continue
    if output == [] and num_tries < 10000:  # pylint: disable=g-explicit-bool-comparison
      # Disallow empty list output unless really necessary.
      continue

    for name in input_names:
      new_inputs_dict[name].extend(inputs_dict[name])
    new_outputs.append(output)

    existing_input_tuples.append(inputs_tuple)
    existing_outputs.append(output)

  return new_inputs_dict, new_outputs


def main():
  if os.path.exists(OUTPUT_FILENAME):
    raise ValueError(f'Output file {OUTPUT_FILENAME} already exists! '
                     'Delete it first to regenerate it.')

  tasks = deepcoder_tasks.HANDWRITTEN_TASKS + deepcoder_tasks.SYNTHETIC_TASKS

  all_results = []
  for task in tasks:
    print(f'Generating for task {task.name}...')
    held_out_inputs_dict, held_out_outputs = generate(task)
    all_results.append({
        'name': task.name,
        'solution': task.solution,
        'held_out_inputs_dict': held_out_inputs_dict,
        'held_out_outputs': held_out_outputs,
    })
  json_str = json_to_str(all_results, indent=4, limit=3)
  with open(OUTPUT_FILENAME, 'w') as f:
    f.write(json_str)


if __name__ == '__main__':
  main()
