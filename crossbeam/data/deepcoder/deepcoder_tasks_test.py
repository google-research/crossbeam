"""Tests for deepcoder_tasks."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.dsl import deepcoder_utils


class DeepcoderTasksTest(parameterized.TestCase):

  def test_num_tasks(self):
    # This test helps us keep track of how many tasks there are.
    self.assertLen(deepcoder_tasks.HANDWRITTEN_TASKS, 100)

  @parameterized.named_parameters(
      *[(task.name, task) for task in deepcoder_tasks.HANDWRITTEN_TASKS])
  def test_task_solution(self, task):
    actual_outputs = deepcoder_utils.run_program(task.solution,
                                                 task.inputs_dict)
    self.assertEqual(actual_outputs, task.outputs)
    if 'original_solution' in task.kwargs:
      actual_outputs = deepcoder_utils.run_program(
          task.kwargs['original_solution'], task.inputs_dict)
      self.assertEqual(actual_outputs, task.outputs)

  def test_tasks_have_unique_names(self):
    names = {task.name for task in deepcoder_tasks.HANDWRITTEN_TASKS}
    self.assertLen(names, len(deepcoder_tasks.HANDWRITTEN_TASKS))
    names_without_prefix = {name.split(':')[1] for name in names}
    self.assertLen(names_without_prefix, len(names))

  @parameterized.named_parameters(
      *[(task.name, task) for task in deepcoder_tasks.HANDWRITTEN_TASKS])
  def test_tasks_name_prefixes(self, task):
    higher_order_funcs = ['map', 'filter', 'count', 'zipwith', 'scanl1']
    solution = task.solution.lower()
    name_prefix = task.name.split(':')[0]
    self.assertIn(name_prefix, higher_order_funcs + ['none', 'multi'])

    if name_prefix == 'none' or name_prefix in higher_order_funcs:
      for func in higher_order_funcs:
        if name_prefix == func:
          self.assertIn(name_prefix + '(', solution)
        else:
          self.assertNotIn(func + '(', solution)
    else:
      self.assertEqual(name_prefix, 'multi')
      count = sum(func + '(' in solution for func in higher_order_funcs)
      self.assertGreater(count, 1)

  @parameterized.named_parameters(
      *[(task.name, task) for task in deepcoder_tasks.HANDWRITTEN_TASKS])
  def test_tasks_good_examples(self, task):
    # A task has exactly 3 examples if the output is a list, or 5 if the output
    # is a scalar.
    num_examples = len(task.outputs)
    if isinstance(task.outputs[0], list):
      self.assertEqual(num_examples, 3)
    else:
      self.assertEqual(type(task.outputs[0]), int)
      self.assertEqual(num_examples, 5)
    # Outputs all have the same type.
    self.assertLen(set(type(o) for o in task.outputs), 1)
    # All lists have length at most 10.
    self.assertLessEqual(
        max((len(o) for o in task.outputs if isinstance(o, list)), default=0),
        10)
    for inputs in task.inputs_dict.values():
      self.assertLessEqual(
          max((len(i) for i in inputs if isinstance(i, list)), default=0), 10)

if __name__ == '__main__':
  absltest.main()
