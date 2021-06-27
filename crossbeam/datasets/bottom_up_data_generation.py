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

import functools
import pickle as cp
import random
import timeit
from absl import app
from absl import flags
import multiprocessing
from tqdm import tqdm
from crossbeam.algorithm import baseline_enumeration
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS


def perform_search(domain, min_weight, max_weight, num_examples, num_inputs,
                   timeout, num_tasks):
  """Generates training data by running bottom-up synthesizer."""

  inputs_dict = domain.inputs_dict_generator(num_inputs=num_inputs,
                                             num_examples=num_examples)
  # Make some dummy outputs. Note that they shouldn't have overlaps with the
  # inputs, or with each other, so that we don't extract unwanted constants.
  assert num_examples <= 4
  dummy_outputs = ['~', '&', '=', '^'][:num_examples]
  task = task_module.Task(inputs_dict, dummy_outputs)

  start_time = timeit.default_timer()
  _, value_set, values_by_weight = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=timeout)
  elapsed_time = timeit.default_timer() - start_time

  if elapsed_time > timeout:
    # If timeout, we didn't consider all ops for the largest weight. To avoid
    # biasing toward ops considered first, throw out everything with the largest
    # weight.
    largest_weight = max(i for i in range(min_weight, max_weight + 1)
                         if values_by_weight[i])
    max_weight = largest_weight - 1
  choices = [v for v in value_set
             if min_weight <= v.weight <= max_weight and
             (domain.output_type is None or v.type == domain.output_type)]
  num_tasks = min(num_tasks, len(choices))
  selected_values = random.sample(choices, k=num_tasks)
  return [task_module.Task(inputs_dict, v.values, solution=v)
          for v in selected_values]


def generate_data(domain, min_weight, max_weight,
                  min_num_examples, max_num_examples,
                  min_num_inputs, max_num_inputs,
                  timeout, num_searches, num_tasks_per_search):
  """Generates and writes data by running multiple searches."""
  tasks = []
  for i in range(num_searches):
    num_examples = random.randint(min_num_examples, max_num_examples)
    num_inputs = random.randint(min_num_inputs, max_num_inputs)
    tasks.extend(perform_search(
        domain, min_weight, max_weight, num_examples, num_inputs, timeout,
        num_tasks=num_tasks_per_search))
    print('Completed search {} of {}'.format(i+1, num_searches))
  return tasks


def datagen_worker(seed,
                   domain, min_weight, max_weight,
                   min_num_examples, max_num_examples,
                   min_num_inputs, max_num_inputs,
                   timeout, num_tasks):
  exp_common.set_global_seed(seed)
  num_examples = random.randint(min_num_examples, max_num_examples)
  num_inputs = random.randint(min_num_inputs, max_num_inputs)
  tasks = perform_search(
    domain, min_weight, max_weight, num_examples, num_inputs, timeout,
    num_tasks=num_tasks)
  return tasks


def main(argv):
  del argv
  exp_common.set_global_seed(FLAGS.data_gen_seed)

  domain = domains.get_domain(FLAGS.domain)
  if FLAGS.num_datagen_proc == 1:
    tasks = generate_data(
        domain,
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        min_num_examples=FLAGS.min_num_examples,
        max_num_examples=FLAGS.max_num_examples,
        min_num_inputs=FLAGS.min_num_inputs,
        max_num_inputs=FLAGS.max_num_inputs,
        timeout=FLAGS.data_gen_timeout,
        num_searches=FLAGS.num_searches,
        num_tasks_per_search=FLAGS.num_tasks)

    if FLAGS.verbose:
      for i, task in enumerate(tasks):
        print('Task #{}: {}'.format(i, task))

    with open(FLAGS.output_file, 'wb') as f:
      cp.dump(tasks, f, cp.HIGHEST_PROTOCOL)
  else:
    pool = multiprocessing.Pool(FLAGS.num_datagen_proc)
    seeds = list(range(FLAGS.data_gen_seed, FLAGS.data_gen_seed + FLAGS.num_searches))
    total_num_tasks = 0
    n_shards = 0
    save_prefix = '.'.join(FLAGS.output_file.split('.')[:-1])
    all_tasks = []
    def save_shard(t_list, n_shards):
      with open(save_prefix + '-%d.pkl' % n_shards, 'wb') as f:
        cp.dump(t_list, f, cp.HIGHEST_PROTOCOL)
      return n_shards + 1
    worker_fun = functools.partial(datagen_worker, domain=domain, 
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        min_num_examples=FLAGS.min_num_examples,
        max_num_examples=FLAGS.max_num_examples,
        min_num_inputs=FLAGS.min_num_inputs,
        max_num_inputs=FLAGS.max_num_inputs,
        timeout=FLAGS.data_gen_timeout,
        num_tasks=FLAGS.num_tasks)
    for local_tasks in tqdm(pool.imap_unordered(worker_fun, seeds), total=len(seeds)):
      total_num_tasks += len(local_tasks)
      all_tasks.extend(local_tasks)
      if len(all_tasks) >= FLAGS.shard_size:
        n_shards = save_shard(all_tasks[:FLAGS.shard_size], n_shards)
        all_tasks = all_tasks[FLAGS.shard_size:]
    if len(all_tasks):
      for i in range(0, len(all_tasks), FLAGS.shard_size):
        n_shards = save_shard(all_tasks[i : i + FLAGS.shard_size], n_shards)
    print('total # generated tasks', total_num_tasks)


if __name__ == '__main__':
  app.run(main)
