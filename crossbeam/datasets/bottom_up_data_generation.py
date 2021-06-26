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
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS


def perform_search(domain, max_weight, min_weight, num_examples, num_inputs,
                   timeout, num_tasks):
  """Generates training data by running bottom-up synthesizer."""

  inputs_dict = domain.inputs_dict_generator(num_inputs=num_inputs,
                                             num_examples=num_examples)
  # Make some dummy outputs. Note that they shouldn't have overlaps with the
  # inputs, or with each other, so that we don't extract unwanted constants.
  assert num_examples <= 4
  dummy_outputs = ['~', '&', '=', '^'][:num_examples]
  task = task_module.Task(inputs_dict, dummy_outputs)

  _, value_set, _ = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=timeout)

  choices = [v for v in value_set
             if min_weight <= v.weight <= max_weight and
             (domain.output_type is None or v.type == domain.output_type)]
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
  exp_common.set_global_seed(FLAGS.data_gen_seed)

  domain = domains.get_domain(FLAGS.domain)
  tasks = generate_data(
      domain,
      max_weight=FLAGS.max_task_weight,
      min_weight=FLAGS.min_task_weight,
      num_examples=FLAGS.num_examples,
      num_inputs=FLAGS.num_inputs,
      timeout=FLAGS.data_gen_timeout,
      num_searches=FLAGS.num_searches,
      num_tasks_per_search=FLAGS.num_tasks)

  if FLAGS.verbose:
    print("generated this many tasks: ",len(tasks))
    for i, task in enumerate(tasks):
      print('Task #{}: {}'.format(i, task))

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
