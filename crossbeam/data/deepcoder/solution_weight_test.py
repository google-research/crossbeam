"""Tests for solution_weight.py."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.data.deepcoder import solution_weight
from crossbeam.dsl import deepcoder_utils


class SolutionWeightTest(parameterized.TestCase):

  def test_solution_weight(self):
    # These examples come from the docstring of solution_weight.py.
    long_solution = (
        'ZipWith(lambda u1, u2: (lambda v1, v2: Subtract(v1, v2))(u2, u1), '
        '        x, y)')
    short_solution = 'ZipWith(lambda u1, u2: Subtract(u2, u1), x, y)'
    self.assertEqual(solution_weight.solution_weight(long_solution), 8)
    self.assertEqual(solution_weight.solution_weight(short_solution), 8)

    long_solution = (
        'Map(lambda u1: (lambda v1: Multiply('
        '    v1, (lambda v1: Square(v1))(v1)))(u1), x)')
    short_solution = 'Map(lambda u1: Multiply(u1, Square(u1)), x)'
    self.assertEqual(solution_weight.solution_weight(long_solution), 8)
    self.assertEqual(solution_weight.solution_weight(short_solution), 8)

  @parameterized.named_parameters(
      *[(task.name, task) for task in deepcoder_tasks.SYNTHETIC_TASKS])
  def test_solution_weight_for_synthetic_tasks(self, task):
    weight = solution_weight.solution_weight(task.solution)
    simplified = deepcoder_utils.simplify(task.solution)
    print(f'task.solution: {task.solution}')
    print(f'simplified: {simplified}')
    self.assertEqual(weight, solution_weight.solution_weight(simplified))
    self.assertTrue(task.name.startswith(f'synthetic:weight_{weight}_'))

if __name__ == '__main__':
  absltest.main()
