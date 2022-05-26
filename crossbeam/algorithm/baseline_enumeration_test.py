"""Tests for baseline_enumeration."""

from absl.testing import absltest

from crossbeam.algorithm import baseline_enumeration
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module


class BaselineEnumerationTest(absltest.TestCase):

  def test_synthesis_works_with_lambdas(self):
    inputs_dict = {
        'delta': [3, 5],
        'lst': [[1, 2, 3], [4, 5, 6]],
    }
    outputs = [
        [4, 5, 6],
        [9, 10, 11],
    ]
    task = task_module.Task(inputs_dict, outputs, solution='')
    domain = domains.get_domain('deepcoder')
    result, _, _, _ = baseline_enumeration.synthesize_baseline(
        task, domain, timeout=5)
    self.assertEqual(result.expression(),
                     'Map(lambda u1: (lambda v1: Add(delta, v1))(u1), lst)')
    self.assertEqual(result.get_weight(), 6)


if __name__ == '__main__':
  absltest.main()
