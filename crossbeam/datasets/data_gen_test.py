"""Tests for crossbeam.datasets.data_gen."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment import exp_common


class DataGenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tuple', 'tuple'),
      ('arithmetic', 'arithmetic'),
      ('bustle', 'bustle'))
  def test_is_deterministic(self, domain_str):
    exp_common.set_global_seed(123)

    domain = domains.get_domain(domain_str)
    generate_tasks_func = functools.partial(
        data_gen.gen_random_tasks,
        domain, num_tasks=5, min_weight=3, max_weight=6,
        min_num_examples=2, max_num_examples=4,
        min_num_inputs=1, max_num_inputs=3, verbose=False)
    tasks = generate_tasks_func()
    self.assertLen(tasks, 5)
    for task in tasks:
      self.assertTrue(1 <= task.num_inputs <= 3)
      self.assertTrue(2 <= task.num_examples <= 4)
      self.assertTrue(3 <= task.solution.get_weight() <= 6)

    exp_common.set_global_seed(123)
    other_tasks = generate_tasks_func()
    for task, other_task in zip(tasks, other_tasks):
      self.assertEqual(str(task), str(other_task))


if __name__ == '__main__':
  absltest.main()
