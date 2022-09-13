"""Tests for deepcoder_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.dsl import deepcoder_utils


class DeepcoderUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_change', 'Multiply(Add(x, y), 3)', None),
      ('no_change_lambda', 'lambda v1: Add(1, v1)', None),
      ('no_change_map', 'Map(lambda u1: fill, list_to_fill)', None),
      ('simple_change',
       'Map(lambda u1: (lambda v1: Add(1, v1))(u1), xs)',
       'Map(lambda u1: Add(1, u1), xs)'),
      ('replace_multiple_in_one_lambda',
       'ZipWith(lambda u1, u2: (lambda v1, v2: Add(v2, Multiply(v1, Add(v2, v1))))(u2, u1), x, y)',
       'ZipWith(lambda u1, u2: Add(u1, Multiply(u2, Add(u1, u2))), x, y)'),
      ('replace_multiple_lambdas',
       'Map(lambda u1: (lambda v1: Add(1, v1))(u1), ZipWith(lambda u1, u2: (lambda v1, v2: Add(v1, v2))(u1, u1), x, y))',
       'Map(lambda u1: Add(1, u1), ZipWith(lambda u1, u2: Add(u1, u1), x, y))'),
      ('replace_nested_lambdas',
       'ZipWith(lambda u1, u2: (lambda v1, v2: Add(Subtract(v1, v2), Sum(Map(lambda u1: (lambda v1: Add(1, v1))(u1), x))))(u2, u1), x, y)',
       'ZipWith(lambda u1, u2: Add(Subtract(u2, u1), Sum(Map(lambda u1: Add(1, u1), x))), x, y)'
       ),
  )
  def test_simplify(self, program, simplified):
    if simplified is None:
      simplified = program
    self.assertEqual(deepcoder_utils.simplify(program), simplified)

  # run_program is tested in crossbeam/data/deepcoder/deepcoder_tasks_test.py,
  # running on all the handwritten tasks and their handwritten solutions.

if __name__ == '__main__':
  absltest.main()
