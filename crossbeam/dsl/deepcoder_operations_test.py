"""Tests for deepcoder_operations."""

import inspect

from absl.testing import absltest

from crossbeam.dsl import deepcoder_operations


class DeepCoderOperationsTest(absltest.TestCase):

  def test_get_operations(self):
    operation_classes = inspect.getmembers(deepcoder_operations,
                                           inspect.isclass)
    actual_set = {type(x) for x in deepcoder_operations.get_operations()}
    expected_set = {x for _, x in operation_classes
                    if x is not deepcoder_operations.DeepCoderOperation}
    self.assertEqual(actual_set, expected_set)

if __name__ == '__main__':
  absltest.main()
