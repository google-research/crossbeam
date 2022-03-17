r"""Create logic tasks that don't appear in training data, separated by size.

python3 -m crossbeam.datasets.bottom_up_logic_test_data \
--min_weight=5 \
--max_weight=19 \
--train_tasks_pkl_file=~/crossbeam/logic_synthesis_10hr/train-tasks.pkl \
--output_file_format=~/crossbeam/logic_by_weight/test-tasks-size-{}.pkl
"""

import os
import random

from absl import app
from absl import flags
import pickle5 as pkl

from crossbeam.algorithm import baseline_enumeration
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module


FLAGS = flags.FLAGS

flags.DEFINE_integer('min_weight', 5,
                     'min weight of tasks to generate')
flags.DEFINE_integer('max_weight', 19,
                     'max weight of tasks to generate')
flags.DEFINE_integer('num_tasks_per_weight', 50,
                     'number of tasks to generate for each weight')
flags.DEFINE_string('train_tasks_pkl_file', None,
                    'file for tasks in training, that we should avoid for test')
flags.DEFINE_string('output_file_format', 'test-tasks-size-{}.pkl',
                    'format for output pkl files with placeholder for weight')


def generate_test_data(min_weight, max_weight, num_tasks_per_weight,
                       train_tasks_pkl_file, output_file_format, seed=0):
  """Generates test data by running the bottom-up synthesizer."""
  random.seed(seed)

  domain = domains.get_domain('logic')
  num_inputs = 4
  num_examples = 1
  inputs_dict = domain.inputs_dict_generator(num_inputs=num_inputs,
                                             num_examples=num_examples)
  # Make some dummy outputs. Note that they shouldn't have overlaps with the
  # inputs, or with each other, so that we don't extract unwanted constants.
  assert num_examples <= 4
  dummy_outputs = ['~', '&', '=', '^'][:num_examples]
  task = task_module.Task(inputs_dict, dummy_outputs)

  _, _, values_by_weight, _ = baseline_enumeration.synthesize_baseline(
      task, domain, max_weight=max_weight, timeout=None)
  print('Finished bottom-up search.\n')

  with open(os.path.expanduser(train_tasks_pkl_file), 'rb') as f:
    training_tasks = pkl.load(f)
  print('Loaded {} training tasks to avoid.'.format(len(training_tasks)))
  training_values_set = set(t.solution for t in training_tasks)
  print('The training tasks contain {} unique values to avoid.'.format(
      len(training_values_set)))

  for weight in range(min_weight, max_weight + 1):
    values_with_weight = list(values_by_weight[weight])
    print('We have {} distinct values of size {}'.format(
        len(values_with_weight), weight))
    random.shuffle(values_with_weight)
    chosen_values = []
    for value in values_with_weight:
      if value in training_values_set:
        continue
      chosen_values.append(value)
      if len(chosen_values) == num_tasks_per_weight:
        break
    print('  Selected {} values that were not in training.'.format(
        len(chosen_values)))

    tasks = [task_module.Task(inputs_dict, v.values, solution=v)
             for v in chosen_values]
    output_filename = os.path.expanduser(output_file_format.format(weight))
    with open(output_filename, 'wb') as f:
      pkl.dump(tasks, f, pkl.HIGHEST_PROTOCOL)
    print('  Wrote output: {}'.format(output_filename))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate_test_data(min_weight=FLAGS.min_weight,
                     max_weight=FLAGS.max_weight,
                     num_tasks_per_weight=FLAGS.num_tasks_per_weight,
                     train_tasks_pkl_file=FLAGS.train_tasks_pkl_file,
                     output_file_format=FLAGS.output_file_format)

if __name__ == '__main__':
  app.run(main)
