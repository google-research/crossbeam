from crossbeam.common import config

import random
import numpy as np
from absl import app
from absl import flags
import pickle as cp
from crossbeam.datasets import random_data
from crossbeam.datasets import bustle_data
from crossbeam.datasets import data_gen_flags
from crossbeam.dsl import domains
from crossbeam.dsl import value as value_module
from crossbeam.experiment import exp_common

FLAGS = flags.FLAGS


def task_gen(domain, min_weight, max_weight, num_examples, num_inputs,
             verbose=False):
  """Generates a random task."""
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
                     min_weight, max_weight, num_examples, num_inputs,
                     verbose=False):
  """Generates multiple random tasks."""
  return [
      task_gen(domain=domain,  # pylint: disable=g-complex-comprehension
               min_weight=min_weight,
               max_weight=max_weight,
               num_examples=num_examples,
               num_inputs=num_inputs,
               verbose=verbose)
      for _ in range(num_tasks)
  ]


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
  exp_common.set_global_seed(FLAGS.seed)

  domain = domains.get_domain(FLAGS.domain)
  eval_tasks = gen_random_tasks(domain,
                                num_tasks=FLAGS.num_tasks,
                                min_weight=FLAGS.min_task_weight,
                                max_weight=FLAGS.max_task_weight,
                                num_examples=FLAGS.num_examples,
                                num_inputs=FLAGS.num_inputs,
                                verbose=FLAGS.verbose)

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
