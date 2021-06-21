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

from crossbeam.datasets import bottom_up_data_generation
from crossbeam.dsl import domains


class BottomUpDataGenerationTest(absltest.TestCase):

  def test_runs(self):
    domain = domains.get_domain('bustle')
    tasks = bottom_up_data_generation.generate_data(
        domain,
        max_weight=8,
        min_weight=5,
        num_examples=3,
        num_inputs=2,
        timeout=5,
        num_searches=2,
        num_tasks_per_search=10)

    self.assertLen(tasks, 20)
    self.assertTrue(all(5 <= t.solution.weight <= 8 for t in tasks))


if __name__ == '__main__':
  absltest.main()
