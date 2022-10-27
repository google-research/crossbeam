# Copyright 2022 Google LLC
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

"""Tests for crossbeam.property_signatures.property_signatures."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.dsl import deepcoder_operations
from crossbeam.dsl import value as value_module
from crossbeam.property_signatures import property_signatures


class CheckerTest(parameterized.TestCase):

  def test_property_signature_value_same_length(self):
    # Construct a variety of Value objects.
    ten = value_module.ConstantValue(10)
    true = value_module.ConstantValue(True)
    x1 = value_module.InputVariable([1, 2, 3], name='x1')
    v1 = value_module.get_free_variable(0)
    v2 = value_module.get_free_variable(1)

    add_op = deepcoder_operations.Add()

    concrete_1 = add_op.apply([ten, x1])
    self.assertEqual(concrete_1.expression(), 'Add(10, x1)')

    lambda_1 = add_op.apply([x1, v1], free_variables=[v1])
    self.assertEqual(lambda_1.expression(), 'lambda v1: Add(x1, v1)')

    lambda_2 = add_op.apply([v2, v1], free_variables=[v1, v2])
    self.assertEqual(lambda_2.expression(), 'lambda v1, v2: Add(v2, v1)')

    bad_lambda = add_op.apply([v1, true], free_variables=[v1])  # Always fails.
    self.assertEqual(bad_lambda.expression(), 'lambda v1: Add(v1, True)')

    # Try a lot of combinations comparing a Value and OutputValue.
    values = [
        ten,
        x1,
        concrete_1,
        lambda_1,
        lambda_2,
        bad_lambda,
        value_module.InputVariable([[1], [2], [3]], 'list_input'),
        value_module.ConstantValue(True),
        # Many examples.
        value_module.InputVariable([1, 2, 3, 4, 5, 6], 'in3'),
        # Few examples.
        value_module.InputVariable([-1, -2], 'in4'),
    ]
    output_values = [
        value_module.OutputValue([7, 8, 9]),
        value_module.OutputValue([True, False, True]),
        value_module.OutputValue([[1, 1], [2, 2], [3, 3]]),
        # Many examples.
        value_module.OutputValue([7, 8, 9, 10, 11, 12]),
        # Few examples.
        value_module.OutputValue([True, False]),
    ]
    lengths = []
    for value, output in itertools.product(values, output_values):
      if (value.num_examples == 1 or output.num_examples == 1 or
          value.num_examples == output.num_examples):
        signature = property_signatures.property_signature_value(
            value, output, fixed_length=True)
        lengths.append(len(signature))
        self.assertTrue(all(len(element) == 2 for element in signature))

    self.assertLen(set(lengths), 1)  # All lengths should be the same.

  def test_property_signature_io_examples_same_length(self):
    # Construct a variety of Value objects.
    input_examples = [
        # 1 input.
        [value_module.InputVariable([1, 2, 3, 4], name='x1')],
        # 2 inputs.
        [value_module.InputVariable([True, False, False, True], name='x1'),
         value_module.InputVariable([[5], [6], [7], [8]], name='x2')],
        # 4 inputs.
        [value_module.InputVariable([1], name='x1'),
         value_module.InputVariable([2], name='x2'),
         value_module.InputVariable([3], name='x3'),
         value_module.InputVariable([4], name='x4')],
    ]
    output_values = [
        value_module.OutputValue([10]),
        value_module.OutputValue([[10]]),
        value_module.OutputValue([True, False, True, False]),
        value_module.OutputValue([[1, 1], [2, 2], [3, 3], [4, 4]]),
    ]
    # Try a lot of combinations.
    lengths = []
    for inputs, output in itertools.product(input_examples, output_values):
      if inputs[0].num_examples == output.num_examples:
        signature = property_signatures.property_signature_io_examples(
            inputs, output, fixed_length=True)
        lengths.append(len(signature))
        self.assertTrue(all(len(element) == 2 for element in signature))

    self.assertLen(set(lengths), 1)  # All lengths should be the same.

if __name__ == '__main__':
  absltest.main()

