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

import json
import os
import pickle5 as cp
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
flags.DEFINE_integer('num_eval_tasks', 5, 'number of evaluation tasks')


def run_synthesis(domain, tasks, timeout, max_values_explored=None,
                  max_weight=20, verbose=False, output_file=None):
  """Performs baseline synthesis on the tasks."""
  num_tasks = len(tasks)
  print('Num tasks: {}'.format(num_tasks))
  success_count = 0
  successful_times = []
  json_dict = {'results': []}

  for i, task in enumerate(tasks):
    start_time = timeit.default_timer()
    result, value_set, _, stats = baseline_enumeration.synthesize_baseline(
        task, domain, max_weight=max_weight, timeout=timeout,
        max_values_explored=max_values_explored)
    elapsed_time = timeit.default_timer() - start_time

    json_dict['results'].append({
        'task': str(task),
        'task_solution': task.solution.expression() if task.solution else None,
        'task_solution_weight': task.solution.get_weight() if task.solution else None,
        'success': bool(result),
        'elapsed_time': elapsed_time,
        'num_values_explored': stats['num_values_explored'],
        'num_unique_values': len(value_set),
        'solution': result.expression() if result else None,
        'solution_weight': result.get_weight() if result else None,
    })

    if verbose:
      print('Task {}: {}'.format(i, task))
      print('Task solution has weight {}'.format(task.solution.get_weight()
                                                 if task.solution else None))
      print('Solution: {}, weight {}'.format(
          result.expression() if result else None,
          result.get_weight() if result else None))
      print('Time: {:.2f}, num values explored: {}, num distinct tasks: {}'
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

  if FLAGS.eval_set_pkl:
    with open(os.path.expanduser(FLAGS.eval_set_pkl), 'rb') as f:
      tasks = cp.load(f)
    if FLAGS.num_tasks > 0:
      tasks = tasks[:FLAGS.num_tasks]
  else:
    tasks = data_gen.gen_random_tasks(domain,
                                      num_tasks=FLAGS.num_eval_tasks,
                                      min_weight=FLAGS.min_task_weight,
                                      max_weight=FLAGS.max_task_weight,
                                      num_examples=FLAGS.num_examples,
                                      num_inputs=FLAGS.num_inputs,
                                      verbose=FLAGS.verbose)

  run_synthesis(domain=domain,
                tasks=tasks,
                timeout=FLAGS.timeout,
                max_values_explored=FLAGS.max_values_explored,
                verbose=FLAGS.verbose,
                output_file=FLAGS.json_results_file)


if __name__ == '__main__':
  app.run(main)
