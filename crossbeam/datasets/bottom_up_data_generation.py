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

"""Generates training data using a bottom-up synthesizer."""

import itertools
import pickle as cp
import random

from absl import app
from absl import flags

from crossbeam.algorithm import baseline_enumeration
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module

FLAGS = flags.FLAGS


def perform_search(domain, max_weight, min_weight, num_examples, num_inputs,
                   timeout, num_tasks):
  """Generates training data by running bottom-up synthesizer."""

  # TODO(kshi): Handle generating inputs in a dependent way
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

  outputs = ['unreachable output {}'.format(i) for i in range(num_examples)]
  task = task_module.Task(inputs_dict, outputs)

  _, _, values_by_weight = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=timeout)

  choices = list(itertools.chain(
      *(values_by_weight[w] for w in range(min_weight, max_weight + 1))))

  num_tasks = min(num_tasks, len(choices))
  selected_values = random.sample(choices, k=num_tasks)
  return [task_module.Task(inputs_dict, v.values, solution=v)
          for v in selected_values]


def generate_data(domain, max_weight, min_weight, num_examples, num_inputs,
                  timeout, num_searches, num_tasks_per_search):
  """Generates and writes data by running multiple searches."""
  tasks = []
  for _ in range(num_searches):
    tasks.extend(perform_search(
        domain, max_weight, min_weight, num_examples, num_inputs, timeout,
        num_tasks=num_tasks_per_search))
  return tasks


def main(argv):
  del argv

  domain = domains.get_domain(FLAGS.domain)
  tasks = generate_data(
      domain,
      max_weight=FLAGS.max_weight,
      min_weight=FLAGS.min_weight,
      num_examples=FLAGS.num_examples,
      num_inputs=FLAGS.num_inputs,
      timeout=FLAGS.data_gen_timeout,
      num_searches=FLAGS.num_searches,
      num_tasks_per_search=FLAGS.num_tasks_per_search)

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
