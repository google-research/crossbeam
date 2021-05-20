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
from crossbeam.datasets import bustle_data
from crossbeam.datasets import data_gen
from crossbeam.datasets import random_data
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_set_pkl', None, 'pkl file for evaluation set tasks')
flags.DEFINE_float('timeout', 5, 'time limit in seconds')
flags.DEFINE_integer('num_tasks', 5, 'number of evaluation tasks')


def main(argv):
  del argv
  exp_common.set_global_seed(FLAGS.seed)

  constants_extractor = None
  if FLAGS.domain == 'tuple':
    constants, operations = data_gen.tuple_consts_and_ops()
    input_generator = random_data.RANDOM_INTEGER
  elif FLAGS.domain == 'arithmetic':
    constants, operations = data_gen.arithmetic_consts_and_ops()
    input_generator = random_data.RANDOM_INTEGER
  elif FLAGS.domain == 'bustle':
    constants, operations = data_gen.bustle_consts_and_ops()
    input_generator = bustle_data.bustle_input_generator
    constants_extractor = bustle_data.bustle_constants_extractor

  if FLAGS.eval_set_pkl:
    with open(FLAGS.eval_set_pkl, 'rb') as f:
      eval_tasks = cp.load(f)
    if FLAGS.num_tasks > 0:
      eval_tasks = eval_tasks[:FLAGS.num_tasks]
  else:
    eval_tasks = [data_gen.task_gen(FLAGS, operations, input_generator,  # pylint: disable=g-complex-comprehension
                                    constants=constants,
                                    constants_extractor=constants_extractor)
                  for _ in range(FLAGS.num_tasks)]

  print('Num tasks: {}'.format(FLAGS.num_tasks))
  success_count = 0
  successful_times = []

  for i, task in enumerate(eval_tasks):
    start_time = timeit.default_timer()
    result, value_set, _ = baseline_enumeration.synthesize_baseline(
        task, operations, max_weight=10, timeout=FLAGS.timeout,
        constants=constants, constants_extractor=constants_extractor)
    elapsed_time = timeit.default_timer() - start_time

    if FLAGS.verbose:
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
      success_count, FLAGS.num_tasks, success_count / FLAGS.num_tasks * 100,
      FLAGS.timeout))
  print('Successful solve time mean: {:.2f} sec, median: {:.2f} sec'.format(
      np.mean(successful_times), np.median(successful_times)))


if __name__ == '__main__':
  app.run(main)
