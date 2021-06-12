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

import pickle as cp
import timeit
from absl import app
from absl import flags
import numpy as np

from crossbeam.algorithm import baseline_enumeration
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_set_pkl', None, 'pkl file for evaluation set tasks')
flags.DEFINE_float('timeout', 5, 'time limit in seconds')
flags.DEFINE_integer('num_tasks', 5, 'number of evaluation tasks')


def run_synthesis(domain, tasks, timeout, verbose=False):
  """Performs baseline synthesis on the tasks."""
  num_tasks = len(tasks)
  print('Num tasks: {}'.format(num_tasks))
  success_count = 0
  successful_times = []
  results_and_times = []

  for i, task in enumerate(tasks):
    start_time = timeit.default_timer()
    result, value_set, _ = baseline_enumeration.synthesize_baseline(
        task, domain, max_weight=10, timeout=timeout)
    elapsed_time = timeit.default_timer() - start_time
    results_and_times.append((result, elapsed_time))

    if verbose:
      print('Task {}: {}'.format(i, task))
      print('Task solution has weight {}'.format(task.solution.weight))
      print('Solution: {}, weight {}'.format(
          result.expression() if result else None,
          result.weight if result else None))
      print('Time: {:.2f}, num values explored: {}'.format(
          elapsed_time, len(value_set)))
      print()

    if result is not None:
      success_count += 1
      successful_times.append(elapsed_time)

  print('Solved {} / {} ({:.2f}%) tasks within {} sec time limit each'.format(
      success_count, num_tasks, success_count / num_tasks * 100, timeout))
  print('Successful solve time mean: {:.2f} sec, median: {:.2f} sec'.format(
      np.mean(successful_times), np.median(successful_times)))

  return results_and_times


def main(argv):
  del argv

  exp_common.set_global_seed(FLAGS.seed)

  domain = domains.get_domain(FLAGS.domain)

  if FLAGS.eval_set_pkl:
    with open(FLAGS.eval_set_pkl, 'rb') as f:
      tasks = cp.load(f)
    if FLAGS.num_tasks > 0:
      tasks = tasks[:FLAGS.num_tasks]
  else:
    tasks = data_gen.gen_random_tasks(domain,
                                      num_tasks=FLAGS.num_tasks,
                                      min_weight=FLAGS.min_task_weight,
                                      max_weight=FLAGS.max_task_weight,
                                      num_examples=FLAGS.num_examples,
                                      num_inputs=FLAGS.num_inputs,
                                      verbose=FLAGS.verbose)

  run_synthesis(domain=domain,
                tasks=tasks,
                timeout=FLAGS.timeout,
                verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
