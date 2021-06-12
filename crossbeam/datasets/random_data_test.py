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

"""Tests for crossbeam.datasets.random_data."""

from absl.testing import absltest

from crossbeam.datasets import random_data
from crossbeam.dsl import domains
from crossbeam.dsl import tuple_operations


class RandomDataTest(absltest.TestCase):

  def test_generate_task_with_index(self):
    inputs_dict = {'a': [1, 2], 'b': [3, 4]}
    constants = [0]
    operations = tuple_operations.get_operations()
    max_weight = 7

    dp_info = random_data.num_expressions_dp(
        operations=operations,
        num_inputs=2,
        constants=constants,
        max_weight=max_weight)

    expected_num_expressions = [0, 3, 0, 3**2, 3**3, 2 * 3**3, 5 * 3**4,
                                5 * 3**4 + 3 * 3**5]
    self.assertEqual(dp_info.num_expressions, expected_num_expressions)
    self.assertEqual(dp_info.answer, sum(expected_num_expressions))

    value_set = set()
    expression_set = set()

    for value_index in range(dp_info.answer):
      value = random_data.generate_value_with_index(
          inputs_dict=inputs_dict,
          constants=constants,
          num_examples=2,
          operations=operations,
          dp_info=dp_info,
          value_index=value_index)
      value_set.add(value)
      expression_set.add(value.expression())
      self.assertLessEqual(value.weight, max_weight)

    self.assertLen(value_set, dp_info.answer)
    self.assertLen(expression_set, dp_info.answer)

  def test_generate_random_task(self):
    domain = domains.TUPLE_DOMAIN
    for max_weight in range(1, 8):
      for min_weight in range(1, max_weight + 1):
        for _ in range(100):
          task = random_data.generate_random_task(
              domain,
              min_weight=min_weight,
              max_weight=max_weight,
              num_examples=3,
              num_inputs=2)
          if max_weight == min_weight == 2:
            # This is the only combination that leaves no possible tasks.
            self.assertIsNone(task)
          else:
            self.assertIn(task.solution.weight,
                          range(min_weight, max_weight + 1))


if __name__ == '__main__':
  absltest.main()
