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
import itertools
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
                   timeout, num_tasks, skip_probability=0,
                   lambda_skip_probability=0, lambda_fraction=None):
  """Generates training data by running bottom-up synthesizer."""
  inputs_dict = domain.inputs_dict_generator(num_inputs=num_inputs,
                                             num_examples=num_examples)
  # Make some dummy outputs. Note that they shouldn't have overlaps with the
  # inputs, or with each other, so that we don't extract unwanted constants.
  assert num_examples <= 5
  dummy_outputs = ['~', '&', '=', '^', '*'][:num_examples]
  task = task_module.Task(inputs_dict, dummy_outputs)

  start_time = timeit.default_timer()
  _, value_set, values_by_weight, _ = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=timeout,
      skip_probability=skip_probability,
      lambda_skip_probability=lambda_skip_probability,
      shuffle_ops=True)
  elapsed_time = timeit.default_timer() - start_time

  largest_weight = max(i for i in range(min_weight, max_weight + 1)
                       if values_by_weight[i])
  print(f'Enumerated programs up to size {largest_weight} in '
        f'{elapsed_time:.2f} seconds.')
  print(f'Considering programs between size {min_weight} and {max_weight} '
        f'inclusive.')

  output_types = (domain.output_type if isinstance(domain.output_type,
                                                   (list, tuple))
                  else [domain.output_type])
  choices = []
  for v in value_set:
    expression = v.expression()
    if (min_weight <= v.get_weight() <= max_weight and
        (domain.output_type is None or v.type in output_types) and
        all(input_name in expression for input_name in inputs_dict)):
      choices.append(v)
  num_choices = len(choices)
  print(f'Found {num_choices} choices.')
  random.shuffle(choices)

  if lambda_fraction is not None:
    assert 0 <= lambda_fraction <= 1
    assert isinstance(num_tasks, int)

    target_num_with_lambda = round(num_tasks * lambda_fraction)
    target_num_without_lambda = num_tasks - target_num_with_lambda
    choices_with_lambda = [v for v in choices if v.contains_lambda]
    choices_without_lambda = [v for v in choices if not v.contains_lambda]
    print(f'{len(choices_with_lambda)} values have lambdas, and '
          f'{len(choices_without_lambda)} values do not.')

    if target_num_with_lambda > len(choices_with_lambda):
      # We don't have enough values with lambdas. Use them all and fill the rest
      # with values without lambdas.
      choices_without_lambda = (
          choices_without_lambda[:num_tasks - len(choices_with_lambda)])
    elif target_num_without_lambda > len(choices_without_lambda):
      # Not enough values without lambdas.
      choices_with_lambda = (
          choices_with_lambda[:num_tasks - len(choices_without_lambda)])
    else:
      # We have enough values of both kinds.
      choices_with_lambda = choices_with_lambda[:target_num_with_lambda]
      choices_without_lambda = choices_without_lambda[:target_num_without_lambda]

    print(f'Choosing {len(choices_with_lambda)} values with lambdas, and '
          f'{len(choices_without_lambda)} values without lambdas.')
    choices = choices_with_lambda + choices_without_lambda
    random.shuffle(choices)

    assert len(choices) <= num_tasks
    if len(choices) < num_tasks:
      assert len(choices) == num_choices  # We used everything we had.

  single_split = isinstance(num_tasks, int)
  if single_split:
    num_tasks = min(num_tasks, len(choices))
    num_tasks = [num_tasks]

  assert isinstance(num_tasks, (list, tuple))
  if sum(num_tasks) > len(choices):
    raise ValueError(
        'Splits sum to {} tasks, but there are only {} values'.format(
            sum(num_tasks), len(choices)))

  splits = []
  last_index = 0
  for split_num_tasks in num_tasks:
    this_split = [task_module.Task(inputs_dict, v.values, solution=v)
                  for v in choices[last_index : last_index + split_num_tasks]]
    assert len(this_split) == split_num_tasks
    last_index += split_num_tasks
    splits.append(this_split)

  if single_split:
    assert len(splits) == 1
    return splits[0]
  else:
    return splits


def generate_data(domain, min_weight, max_weight,
                  min_num_examples, max_num_examples,
                  min_num_inputs, max_num_inputs,
                  timeout, num_searches, num_tasks_per_search,
                  skip_probability=0, lambda_skip_probability=0,
                  lambda_fraction=None):
  """Generates and writes data by running multiple searches."""
  tasks = []
  for i in range(num_searches):
    num_examples = random.randint(min_num_examples, max_num_examples)
    num_inputs = random.randint(min_num_inputs, max_num_inputs)
    tasks.extend(perform_search(
        domain, min_weight, max_weight, num_examples, num_inputs, timeout,
        num_tasks=num_tasks_per_search, skip_probability=skip_probability,
        lambda_skip_probability=lambda_skip_probability,
        lambda_fraction=lambda_fraction))
    print('Completed search {} of {}. {} tasks total.'.format(i+1, num_searches, len(tasks)))
  return tasks


def datagen_worker(seed,
                   domain, min_weight, max_weight,
                   min_num_examples, max_num_examples,
                   min_num_inputs, max_num_inputs,
                   timeout, num_tasks, skip_probability=0,
                   lambda_skip_probability=0, lambda_fraction=None):
  exp_common.set_global_seed(seed)
  num_examples = random.randint(min_num_examples, max_num_examples)
  num_inputs = random.randint(min_num_inputs, max_num_inputs)
  tasks = perform_search(
      domain, min_weight, max_weight, num_examples, num_inputs, timeout,
      num_tasks=num_tasks, skip_probability=skip_probability,
      lambda_skip_probability=lambda_skip_probability,
      lambda_fraction=lambda_fraction)
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
        num_tasks_per_search=FLAGS.num_tasks,
        skip_probability=FLAGS.skip_probability,
        lambda_skip_probability=FLAGS.lambda_skip_probability,
        lambda_fraction=FLAGS.lambda_fraction)

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
      with open(save_prefix + '-%05d.pkl' % n_shards, 'wb') as f:
        cp.dump(t_list, f, cp.HIGHEST_PROTOCOL)
      return n_shards + 1
    worker_fun = functools.partial(
        datagen_worker, domain=domain,
        min_weight=FLAGS.min_task_weight,
        max_weight=FLAGS.max_task_weight,
        min_num_examples=FLAGS.min_num_examples,
        max_num_examples=FLAGS.max_num_examples,
        min_num_inputs=FLAGS.min_num_inputs,
        max_num_inputs=FLAGS.max_num_inputs,
        timeout=FLAGS.data_gen_timeout,
        num_tasks=FLAGS.num_tasks,
        skip_probability=FLAGS.skip_probability,
        lambda_skip_probability=FLAGS.lambda_skip_probability,
        lambda_fraction=FLAGS.lambda_fraction)
    for local_tasks in tqdm(pool.imap_unordered(worker_fun, seeds), total=len(seeds)):
      total_num_tasks += len(local_tasks)
      all_tasks.extend(local_tasks)
      while len(all_tasks) >= FLAGS.shard_size:
        n_shards = save_shard(all_tasks[:FLAGS.shard_size], n_shards)
        all_tasks = all_tasks[FLAGS.shard_size:]
    if len(all_tasks):
      save_shard(all_tasks, n_shards)
    print('total # generated tasks', total_num_tasks)


if __name__ == '__main__':
  app.run(main)
