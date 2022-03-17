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

"""Tests for crossbeam.datasets.bustle_data."""


from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.datasets import bustle_data
from crossbeam.dsl import task as task_module


class BustleDataTest(parameterized.TestCase):

  @parameterized.parameters(
      ('computer programs are cool', 'dynamic programming is neat', ' program'),
      ('123456', '1356234', '234'),
      ('aaaaa', 'a', 'a'),
      ('exact match', 'exact match', 'exact match'),
      ('aaaaa', 'b', ''),
      ('', '', ''))
  def test_compute_lcs(self, x, y, expected):
    self.assertEqual(bustle_data.compute_lcs(x, y), expected)

  def test_bustle_constants_extractor(self):
    inputs_dict = {'input_1': ['abcd', 'wxyz', '0ab12'],
                   'input_2': ['$1234', '$5.67', '$89']}
    outputs = ['01/01/01', '11/30/98', '']
    task = task_module.Task(inputs_dict, outputs)

    old = bustle_data.COMMON_CONSTANTS
    bustle_data.COMMON_CONSTANTS = [
        '.',  # Appears once for input_2.
        '#',  # Doesn't appear anywhere.
        '/']  # Appears in the output as a substring of another constant.

    expected = bustle_data.ALWAYS_USED_CONSTANTS + [
        'ab',  # LCS from input_1.
        '1/',  # LCS from output.
        '.',  # Common constant appearing in input_2.
        '/',  # Common constant appearing in output.
    ]
    # "$" appears in every instance of input_2, but it is only 1 character, and
    # isn't in the common constants list, so it is not extracted.
    self.assertCountEqual(bustle_data.bustle_constants_extractor(task),
                          expected)

    bustle_data.COMMON_CONSTANTS = old


if __name__ == '__main__':
  absltest.main()
