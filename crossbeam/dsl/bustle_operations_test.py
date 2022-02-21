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

"""Tests for crossbeam.dsl.bustle_operations."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.dsl import bustle_operations


class BustleOperationsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('Add', [3, 5], 8),

      ('Concatenate', ['a', 'bc'], 'abc'),

      ('Find2', ['b', 'abc'], 2),
      ('Find2', ['B', 'abc'], None),

      ('Find3', ['b', 'abcb', 0], None),
      ('Find3', ['b', 'abcb', 1], 2),
      ('Find3', ['b', 'abcb', 2], 2),
      ('Find3', ['b', 'abcb', 3], 4),
      ('Find3', ['b', 'abcb', 4], 4),
      ('Find3', ['b', 'abcb', 5], None),
      ('Find3', ['B', 'abcb', 2], None),

      ('Left', ['abc', -1], None),
      ('Left', ['abc', 0], ''),
      ('Left', ['abc', 2], 'ab'),
      ('Left', ['abc', 3], 'abc'),
      ('Left', ['abc', 4], 'abc'),
      ('Left', ['', 2], ''),

      ('Len', [''], 0),
      ('Len', ['abc'], 3),

      ('Mid', ['abc', 0, 1], None),
      ('Mid', ['abc', 1, -1], None),
      ('Mid', ['abc', 1, 0], ''),
      ('Mid', ['abc', 1, 1], 'a'),
      ('Mid', ['abc', 1, 9], 'abc'),
      ('Mid', ['abc', 2, 0], ''),
      ('Mid', ['abc', 2, 1], 'b'),
      ('Mid', ['abc', 2, 2], 'bc'),
      ('Mid', ['abc', 3, 2], 'c'),
      ('Mid', ['abc', 9, 9], ''),

      ('Replace', ['abc', 0, 1, 'XY'], None),
      ('Replace', ['abc', 1, -1, 'XY'], None),
      ('Replace', ['abc', 1, 0, 'XY'], 'XYabc'),
      ('Replace', ['abc', 1, 1, 'XY'], 'XYbc'),
      ('Replace', ['abc', 1, 3, 'XY'], 'XY'),
      ('Replace', ['abc', 1, 4, 'XY'], 'XY'),
      ('Replace', ['abc', 2, 1, 'XY'], 'aXYc'),
      ('Replace', ['abc', 3, 0, 'XY'], 'abXYc'),
      ('Replace', ['abc', 4, 0, 'XY'], 'abcXY'),
      ('Replace', ['abc', 5, 0, 'XY'], 'abcXY'),
      ('Replace', ['abc', 9, 9, 'XY'], 'abcXY'),

      ('Right', ['abc', -1], None),
      ('Right', ['abc', 0], ''),
      ('Right', ['abc', 1], 'c'),
      ('Right', ['abc', 2], 'bc'),
      ('Right', ['abc', 3], 'abc'),
      ('Right', ['abc', 4], 'abc'),
      ('Right', ['', 2], ''),

      ('Trim', [''], ''),
      ('Trim', ['   '], ''),
      ('Trim', ['    a    '], 'a'),
      ('Trim', ['abc'], 'abc'),
      ('Trim', [' abc'], 'abc'),
      ('Trim', ['abc '], 'abc'),
      ('Trim', [' abc '], 'abc'),
      ('Trim', ['    a   b    c  '], 'a b c'),

      ('Lower', ['abc dEf XYZ 123'], 'abc def xyz 123'),

      ('Upper', ['abc dEf XYZ 123'], 'ABC DEF XYZ 123'),

      ('Proper', ['abc dEf XYZ 123'], 'Abc Def Xyz 123'),
      ('Proper', ['a 1a a1 a1a1a .a?a'], 'A 1A A1 A1A1A .A?A'),

      ('Rept', ['abc', -1], None),
      ('Rept', ['abc', 0], ''),
      ('Rept', ['abc', 1], 'abc'),
      ('Rept', ['abc', 2], 'abcabc'),
      ('Rept', ['abc', 3], 'abcabcabc'),

      ('Substitute3', ['Spreadsheet', 'e', 'E'], 'SprEadshEEt'),
      ('Substitute3', ['Spreadsheet', 'x', 'E'], 'Spreadsheet'),
      ('Substitute3', ['Spreadsheet', '', 'X'], 'Spreadsheet'),
      ('Substitute3', ['', '', 'X'], ''),
      ('Substitute3', ['Spreadsheet', 'e', ''], 'Spradsht'),
      ('Substitute3', ['AAAAA', 'AA', 'X'], 'XXA'),
      ('Substitute3', ['AAAAA', 'AA', 'AXA'], 'AXAAXAA'),
      ('Substitute3', ['AAAAA', '', 'B'], 'AAAAA'),

      ('Substitute4', ['Google Docs', 'ogle', 'od', 1], 'Good Docs'),
      ('Substitute4', ['Google Docs', 'o', 'a', 3], 'Google Dacs'),
      ('Substitute4', ['Spreadsheet', 'e', 'E', 2], 'SpreadshEet'),
      ('Substitute4', ['AAAAA', 'AA', 'X', -1], None),
      ('Substitute4', ['AAAAA', 'AA', 'X', 0], 'XXA'),
      ('Substitute4', ['AAAAA', 'AA', 'X', 1], 'XAAA'),
      ('Substitute4', ['AAAAA', 'AA', 'X', 2], 'AXAA'),
      ('Substitute4', ['AAAAA', 'AA', 'X', 3], 'AAXA'),
      ('Substitute4', ['AAAAA', 'AA', 'X', 4], 'AAAX'),
      ('Substitute4', ['AAAAA', 'AA', 'X', 5], 'AAAAA'),
      ('Substitute4', ['AAAAA', 'AA', 'AXA', 0], 'AXAAXAA'),
      ('Substitute4', ['AAAAA', 'AA', 'AXA', 1], 'AXAAAA'),
      ('Substitute4', ['AAAAA', 'AA', 'AXA', 4], 'AAAAXA'),
      ('Substitute4', ['AAAAA', 'AA', 'AXA', 5], 'AAAAA'),
      ('Substitute4', ['A', '', 'B', -1], None),

      ('ToText', [-99], '-99'),
      ('ToText', [0], '0'),
      ('ToText', [123], '123'),

      ('If', [True, 'A', 'B'], 'A'),
      ('If', [False, 'A', 'B'], 'B'),

      ('Exact', ['A', 'A'], True),
      ('Exact', ['A', 'B'], False),

      ('Gt', [5, 6], False),
      ('Gt', [6, 6], False),
      ('Gt', [7, 6], True),

      ('Gte', [5, 6], False),
      ('Gte', [6, 6], True),
      ('Gte', [7, 6], True),
  )
  def test_operations(self, operation_name, arguments, expected_result):
    operation = getattr(bustle_operations, operation_name)()
    self.assertEqual(operation.arg_types(), tuple(type(a) for a in arguments))
    self.assertLen(arguments, operation.arity)

    if expected_result is None:
      with self.assertRaises(ValueError):
        operation.apply_single(arguments)
    else:
      self.assertEqual(operation.apply_single(arguments), expected_result)

  def test_get_operations(self):
    operation_classes = inspect.getmembers(bustle_operations, inspect.isclass)
    actual_set = {type(x) for x in bustle_operations.get_operations()}
    expected_set = {x for _, x in operation_classes
                    if x is not bustle_operations.BustleOperation}
    self.assertEqual(actual_set, expected_set)

if __name__ == '__main__':
  absltest.main()
