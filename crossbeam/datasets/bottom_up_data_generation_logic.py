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

"""Bottom up data generation for the logic domain.

In the logic domain, the inputs are always the same, so we do not need to
perform multiple searches. Every search will produce the same results.
Furthermore, this means that if we generate train/valid/test sets separately, we
may get overlaps between the sets. In order to avoid overlaps, we must split the
results from one search into these sets."""

import pickle as cp
from absl import app
from absl import flags
from crossbeam.datasets import bottom_up_data_generation
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS

flags.DEFINE_multi_integer('num_tasks_per_split', None,
                           'The number of tasks for each split')
flags.DEFINE_multi_string('split_filenames', None, 'Filenames for each split')


def main(argv):
  del argv
  exp_common.set_global_seed(FLAGS.data_gen_seed)

  assert len(FLAGS.num_tasks_per_split) == len(FLAGS.split_filenames)
  assert FLAGS.min_num_inputs == FLAGS.max_num_inputs == 4
  assert FLAGS.min_num_examples == FLAGS.max_num_examples

  domain = domains.get_domain('logic')
  splits = bottom_up_data_generation.perform_search(
      domain=domain,
      min_weight=FLAGS.min_task_weight,
      max_weight=FLAGS.max_task_weight,
      num_examples=FLAGS.min_num_examples,
      num_inputs=FLAGS.min_num_inputs,
      timeout=FLAGS.data_gen_timeout,
      num_tasks=FLAGS.num_tasks_per_split)

  for split_filename, split in zip(FLAGS.split_filenames, splits):
    with open(split_filename, 'wb') as f:
      cp.dump(split, f, cp.HIGHEST_PROTOCOL)
    print('Wrote {} tasks to {}'.format(len(split), split_filename))


if __name__ == '__main__':
  app.run(main)
