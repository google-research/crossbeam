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

"""Tests for crossbeam.datasets.bottom_up_data_generation."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.datasets import bottom_up_data_generation
from crossbeam.dsl import domains


class BottomUpDataGenerationTest(parameterized.TestCase):

  @parameterized.parameters(
      ('tuple',),
      # ('arithmetic',),  # Can't get tasks of even size.
      ('bustle',),
      ('deepcoder',))
  def test_runs(self, domain_str):
    domain = domains.get_domain(domain_str)
    min_weight = 4
    max_weight = 6
    tasks_by_weight = bottom_up_data_generation.generate_data(
        domain,
        min_weight=min_weight,
        max_weight=max_weight,
        min_num_examples=2,
        max_num_examples=4,
        min_num_inputs=1,
        max_num_inputs=2,
        timeout=5,
        num_searches=2,
        num_tasks_per_weight=10)

    for weight in range(min_weight, max_weight + 1):
      tasks = tasks_by_weight[weight]
      # Sometimes a search will find few tasks of a certain weight, because
      # each task must use all inputs at least once which may be difficult.
      self.assertGreaterEqual(len(tasks), 10)

      self.assertTrue(all(t.solution.get_weight() == weight for t in tasks))
      if domain.output_type:
        for t in tasks:
          if isinstance(domain.output_type, (tuple, list)):
            self.assertIn(t.solution.type, domain.output_type)
          else:
            self.assertEqual(t.solution.type, domain.output_type)

if __name__ == '__main__':
  absltest.main()
