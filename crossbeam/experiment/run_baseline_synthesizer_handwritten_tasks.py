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

import functools
import json
import multiprocessing
import os
import timeit

from absl import app
from absl import flags
from crossbeam.algorithm import baseline_enumeration
from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment import exp_common
import numpy as np
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_process', 16, 'number of evaluation process')


def baseline_enumeration_with_timing(task, *args, **kargs):
  start_time = timeit.default_timer()
  result, value_set, _, stats = baseline_enumeration.synthesize_baseline(
      task, *args, **kargs)
  elapsed_time = timeit.default_timer() - start_time
  return task, result, value_set, stats, elapsed_time


def run_synthesis(domain, tasks, timeout, max_values_explored=None,
                  max_weight=20, verbose=False, output_file=None):
  """Performs baseline synthesis on the tasks."""
  num_tasks = len(tasks)
  print('Num tasks: {}'.format(num_tasks))
  success_count = 0
  successful_times = []
  json_dict = {'results': []}

  worker_fun = functools.partial(
      baseline_enumeration_with_timing,
      domain=domain,
      timeout=timeout,
      max_weight=max_weight,
      max_values_explored=max_values_explored)

  pool = multiprocessing.Pool(FLAGS.num_process)
  for task, result, value_set, stats, elapsed_time in tqdm(
      pool.imap_unordered(worker_fun, tasks), total=len(tasks)):
    json_dict['results'].append({
        'task': str(task),
        'success': bool(result),
        'elapsed_time': elapsed_time,
        'num_values_explored': stats['num_values_explored'],
        'num_unique_values': len(value_set),
        'solution': result.expression() if result else None,
        'solution_weight': result.get_weight() if result else None,
    })
    if verbose:
      print('Task: {}'.format(task))
      print('Solution: {}, weight {}'.format(
          result.expression() if result else None,
          result.get_weight() if result else None))
      print('Time: {:.2f}, num values explored: {}, num distinct values: {}'
            .format(elapsed_time, stats['num_values_explored'], len(value_set)))
      print()

    if result is not None:
      success_count += 1
      successful_times.append(elapsed_time)

  json_dict['num_tasks'] = len(tasks)
  json_dict['num_tasks_solved'] = success_count
  json_dict['success_rate'] = success_count / len(tasks)

  print('Solved {} / {} ({:.2f}%) tasks within {} sec time limit or {} values explored'.format(
      success_count, num_tasks, success_count / num_tasks * 100, timeout, max_values_explored))
  print('Successful solve time mean: {:.2f} sec, median: {:.2f} sec'.format(
      np.mean(successful_times), np.median(successful_times)))

  if output_file:
    with open(os.path.expanduser(output_file), 'w') as f:
      json.dump(json_dict, f, indent=4, sort_keys=True)
    print('Wrote JSON results file at {}'.format(output_file))

  return json_dict


def main(argv):
  del argv

  exp_common.set_global_seed(FLAGS.seed)

  domain = domains.get_domain(FLAGS.domain)

  tasks = deepcoder_tasks.HANDWRITTEN_TASKS
  run_synthesis(domain=domain,
                tasks=tasks,
                timeout=FLAGS.timeout,
                max_values_explored=FLAGS.max_values_explored,
                verbose=FLAGS.verbose,
                output_file=FLAGS.json_results_file)


if __name__ == '__main__':
  app.run(main)
