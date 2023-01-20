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

import collections
import functools
import multiprocessing
import os
import pickle as cp
import random
import timeit

from absl import app
from absl import flags
from tqdm import tqdm

from crossbeam.algorithm import baseline_enumeration
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS


def perform_search(domain, min_weight, max_weight, num_examples, num_inputs,
                   timeout, num_tasks_per_weight, skip_probability=0,
                   lambda_skip_probability=0, lambda_fraction=None,
                   shuffle_ops=False):
  """Generates training data by running bottom-up synthesizer."""
  inputs_dict = domain.inputs_dict_generator(num_inputs=num_inputs,
                                             num_examples=num_examples)
  # Make some dummy outputs. Note that they shouldn't have overlaps with the
  # inputs, or with each other, so that we don't extract unwanted constants.
  assert num_examples <= 5
  dummy_outputs = ['~', '&', '=', '^', '*'][:num_examples]
  task = task_module.Task(inputs_dict, dummy_outputs)

  start_time = timeit.default_timer()
  _, _, values_by_weight, _ = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=timeout,
      skip_probability=skip_probability,
      lambda_skip_probability=lambda_skip_probability,
      shuffle_ops=shuffle_ops)
  elapsed_time = timeit.default_timer() - start_time

  largest_weight = max(i for i in range(min_weight, max_weight + 1)
                       if values_by_weight[i])
  max_weight = largest_weight
  print(f'Enumerated programs up to size {largest_weight} in '
        f'{elapsed_time:.2f} seconds.')

  if shuffle_ops and elapsed_time > timeout:
    # If timeout, we didn't finish enumerating the largest weight. We want to
    # avoid biasing toward ops considered first.
    # We can keep values of the largest weight if we found some lambdas with
    # that weight (which means we finished enumerating non-lambdas first).
    if any(v.num_free_variables > 0 for v in values_by_weight[largest_weight]):
      max_weight = largest_weight
    else:
      # We didn't finish enumerating non-lambdas. Throw out everything with the
      # largest weight.
      max_weight = largest_weight - 1
    print('Reached timeout. Enumerated programs up to size', max_weight)

  print(f'Considering programs between size {min_weight} and {max_weight} '
        f'inclusive.')

  output_types = (domain.output_type if isinstance(domain.output_type,
                                                   (list, tuple))
                  else [domain.output_type])

  tasks_by_weight = {}

  for weight in range(min_weight, max_weight + 1):
    choices = []
    for v in values_by_weight[weight]:
      expression = v.expression()
      if ((domain.output_type is None or v.type in output_types) and
          all(input_name in expression for input_name in inputs_dict)):
        choices.append(v)
    num_choices = len(choices)
    print(f'Found {num_choices} choices for weight {weight}.')
    random.shuffle(choices)

    if lambda_fraction is not None:
      assert 0 <= lambda_fraction <= 1
      assert isinstance(num_tasks_per_weight, int)

      target_num_with_lambda = round(num_tasks_per_weight * lambda_fraction)
      target_num_without_lambda = num_tasks_per_weight - target_num_with_lambda
      choices_with_lambda = [v for v in choices if v.contains_lambda]
      choices_without_lambda = [v for v in choices if not v.contains_lambda]
      print(f'{len(choices_with_lambda)} values have lambdas, and '
            f'{len(choices_without_lambda)} values do not.')

      if target_num_with_lambda > len(choices_with_lambda):
        # We don't have enough values with lambdas. Use them all and fill the
        # rest with values without lambdas.
        choices_without_lambda = choices_without_lambda[
            : num_tasks_per_weight - len(choices_with_lambda)]
      elif target_num_without_lambda > len(choices_without_lambda):
        # Not enough values without lambdas.
        choices_with_lambda = choices_with_lambda[
            : num_tasks_per_weight - len(choices_without_lambda)]
      else:
        # We have enough values of both kinds.
        choices_with_lambda = choices_with_lambda[:target_num_with_lambda]
        choices_without_lambda = choices_without_lambda[
            :target_num_without_lambda]

      print(f'Choosing {len(choices_with_lambda)} values with lambdas, and '
            f'{len(choices_without_lambda)} values without lambdas.')
      choices = choices_with_lambda + choices_without_lambda
      random.shuffle(choices)

      assert len(choices) <= num_tasks_per_weight
      if len(choices) < num_tasks_per_weight:
        assert len(choices) == num_choices  # We used everything we had.

    tasks_by_weight[weight] = [
        task_module.Task(inputs_dict, v.values, solution=v)
        for v in choices]

  return tasks_by_weight


def generate_data(domain, min_weight, max_weight,
                  min_num_examples, max_num_examples,
                  min_num_inputs, max_num_inputs,
                  timeout, num_searches, num_tasks_per_weight,
                  skip_probability=0, lambda_skip_probability=0,
                  lambda_fraction=None, shuffle_ops=False):
  """Generates and writes data by running multiple searches."""
  tasks_by_weight = collections.defaultdict(list)
  num_total_tasks = 0
  for i in range(num_searches):
    num_examples = random.randint(min_num_examples, max_num_examples)
    num_inputs = random.randint(min_num_inputs, max_num_inputs)
    new_tasks_by_weight = perform_search(
        domain, min_weight, max_weight, num_examples, num_inputs, timeout,
        num_tasks_per_weight=num_tasks_per_weight,
        skip_probability=skip_probability,
        lambda_skip_probability=lambda_skip_probability,
        lambda_fraction=lambda_fraction,
        shuffle_ops=shuffle_ops)
    for weight, new_tasks in new_tasks_by_weight.items():
      tasks_by_weight[weight].extend(new_tasks)
      num_total_tasks += len(new_tasks)
    print(f'Completed search {i+1} of {num_searches}. '
          f'{num_total_tasks} tasks total.')
  return tasks_by_weight


def datagen_worker(seed,
                   domain, min_weight, max_weight,
                   min_num_examples, max_num_examples,
                   min_num_inputs, max_num_inputs,
                   timeout, num_tasks_per_weight, skip_probability=0,
                   lambda_skip_probability=0, lambda_fraction=None,
                   shuffle_ops=False):
  """A job to run in parallel using a multiprocessing pool."""
  exp_common.set_global_seed(seed)
  num_examples = random.randint(min_num_examples, max_num_examples)
  num_inputs = random.randint(min_num_inputs, max_num_inputs)
  tasks_by_weight = perform_search(
      domain, min_weight, max_weight, num_examples, num_inputs, timeout,
      num_tasks_per_weight=num_tasks_per_weight,
      skip_probability=skip_probability,
      lambda_skip_probability=lambda_skip_probability,
      lambda_fraction=lambda_fraction,
      shuffle_ops=shuffle_ops)
  return tasks_by_weight


def main(argv):
  del argv
  exp_common.set_global_seed(FLAGS.data_gen_seed)
  try:
    os.makedirs(FLAGS.data_save_dir)
  except FileExistsError:
    pass  # If we check if the path exists, we can still have a race condition.

  domain = domains.get_domain(FLAGS.domain)
  if FLAGS.num_datagen_proc == 1:
    tasks_by_weight = generate_data(
        domain,
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        min_num_examples=FLAGS.min_num_examples,
        max_num_examples=FLAGS.max_num_examples,
        min_num_inputs=FLAGS.min_num_inputs,
        max_num_inputs=FLAGS.max_num_inputs,
        timeout=FLAGS.data_gen_timeout,
        num_searches=FLAGS.num_searches,
        num_tasks_per_weight=FLAGS.num_tasks_per_weight,
        skip_probability=FLAGS.skip_probability,
        lambda_skip_probability=FLAGS.lambda_skip_probability,
        lambda_fraction=FLAGS.lambda_fraction,
        shuffle_ops=FLAGS.shuffle_ops)

    if FLAGS.verbose:
      for weight in sorted(tasks_by_weight.keys()):
        for i, task in enumerate(tasks_by_weight[weight]):
          print(f'Task #{i} for weight {weight}: {task}')

    for weight in sorted(tasks_by_weight.keys()):
      filename = os.path.join(FLAGS.data_save_dir,
                              f'{FLAGS.split}-weight-{weight}.pkl')
      with open(filename, 'wb') as f:
        cp.dump(tasks_by_weight[weight], f, cp.HIGHEST_PROTOCOL)

  else:

    def save_shard(task_list, weight, shard_index):
      filename = os.path.join(
          FLAGS.data_save_dir,
          f'{FLAGS.split}-weight-{weight}-{shard_index:05d}.pkl')
      with open(filename, 'wb') as f:
        cp.dump(task_list, f, cp.HIGHEST_PROTOCOL)

    worker_fun = functools.partial(
        datagen_worker, domain=domain,
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        min_num_examples=FLAGS.min_num_examples,
        max_num_examples=FLAGS.max_num_examples,
        min_num_inputs=FLAGS.min_num_inputs,
        max_num_inputs=FLAGS.max_num_inputs,
        timeout=FLAGS.data_gen_timeout,
        num_tasks_per_weight=FLAGS.num_tasks_per_weight,
        skip_probability=FLAGS.skip_probability,
        lambda_skip_probability=FLAGS.lambda_skip_probability,
        lambda_fraction=FLAGS.lambda_fraction,
        shuffle_ops=FLAGS.shuffle_ops)

    pool = multiprocessing.Pool(FLAGS.num_datagen_proc)
    seeds = list(range(FLAGS.data_gen_seed,
                       FLAGS.data_gen_seed + FLAGS.num_searches))
    shard_index_per_weight = {
        weight: FLAGS.shard_start_index
        for weight in range(FLAGS.min_task_weight, FLAGS.max_task_weight + 1)
    }
    tasks_by_weight = collections.defaultdict(list)
    total_num_tasks = 0

    for local_tasks_by_weight in tqdm(pool.imap_unordered(worker_fun, seeds),
                                      total=len(seeds)):
      for weight in sorted(local_tasks_by_weight.keys()):
        local_tasks = local_tasks_by_weight[weight]
        total_num_tasks += len(local_tasks)
        tasks_by_weight[weight].extend(local_tasks)
        while len(tasks_by_weight[weight]) >= FLAGS.shard_size:
          save_shard(task_list=tasks_by_weight[weight][:FLAGS.shard_size],
                     weight=weight,
                     shard_index=shard_index_per_weight[weight])
          shard_index_per_weight[weight] += 1
          tasks_by_weight[weight] = tasks_by_weight[weight][FLAGS.shard_size:]

    for weight in sorted(tasks_by_weight.keys()):
      if tasks_by_weight[weight]:
        save_shard(task_list=tasks_by_weight[weight],
                   weight=weight,
                   shard_index=shard_index_per_weight[weight])

    print(f'total # generated tasks: {total_num_tasks}')


if __name__ == '__main__':
  app.run(main)
