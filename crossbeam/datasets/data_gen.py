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

import random
from absl import app
from absl import flags
import pickle as cp
from crossbeam.datasets import random_data
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.dsl import value as value_module
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS


def task_gen(domain, min_weight, max_weight,
             min_num_examples, max_num_examples,
             min_num_inputs, max_num_inputs,
             verbose=False):
  """Generates a random task."""
  num_examples = random.randint(min_num_examples, max_num_examples)
  num_inputs = random.randint(min_num_inputs, max_num_inputs)
  task = random_data.generate_good_random_task(
      domain=domain,
      min_weight=min_weight,
      max_weight=max_weight,
      num_examples=num_examples,
      num_inputs=num_inputs)
  if verbose:
    print(task)
  return task


def gen_random_tasks(domain, num_tasks,
                     min_weight, max_weight, min_num_examples, max_num_examples,
                     min_num_inputs, max_num_inputs, verbose=False):
  """Generates multiple random tasks."""
  tasks = []
  for _ in range(num_tasks):
    task = task_gen(domain=domain,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    min_num_examples=min_num_examples,
                    max_num_examples=max_num_examples,
                    min_num_inputs=min_num_inputs,
                    max_num_inputs=max_num_inputs,
                    verbose=verbose)
    tasks.append(task)
  return tasks


def trace_gen(value_node, result=None):
  """Generates a trace for the given OperationValue."""
  if result is None:
    result = []
  if isinstance(value_node, value_module.OperationValue):  # non-leaf
    sub_ops = [arg for arg in value_node.arg_values
               if isinstance(arg, value_module.OperationValue)]
    random.shuffle(sub_ops)
    for child in sub_ops:
      trace_gen(child, result)
    if value_node not in result:
      result.append(value_node)
  return result


def main(argv):
  del argv
  exp_common.set_global_seed(FLAGS.data_gen_seed)

  domain = domains.get_domain(FLAGS.domain)

  if FLAGS.domain != 'logic':
    eval_tasks = gen_random_tasks(domain,
                                  num_tasks=FLAGS.num_tasks,
                                  min_weight=FLAGS.min_task_weight,
                                  max_weight=FLAGS.max_task_weight,
                                  min_num_examples=FLAGS.min_num_examples,
                                  max_num_examples=FLAGS.max_num_examples,
                                  min_num_inputs=FLAGS.min_num_inputs,
                                  max_num_inputs=FLAGS.max_num_inputs,
                                  verbose=FLAGS.verbose)
  else:
    from crossbeam.datasets.logic_data import all_manual_logic_tasks
    operations = domain.operations
    eval_tasks = all_manual_logic_tasks(operations)
    
    #eval_tasks.extend([make_connected_task(operations,p=p) for p in [0.05,0.4]])

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
