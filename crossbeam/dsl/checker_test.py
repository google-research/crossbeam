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

"""Tests for crossbeam.datasets.checker."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.dsl import checker
from crossbeam.dsl import task as task_module


class CheckerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('good_solution', '((a, 0), b)', True),
      ('ok_whitespace', ' ( ( a , 0 ) , b ) ', True),
      ('ok_arithmetic', '((a, 3 - 2 - 1), b)', True),
      ('no_builtins', '((a, int(0)), b)', False),
      ('bad_syntax', '(a, 0), b)', False),
      ('wrong_tuple', '((a, 1), b)', False),
      ('name_error', '((a, 0), c)', False),
  )
  def test_check_solutions_tuple(self, solution_string, expected):
    inputs_dict = {'a': [2, 1, 5], 'b': [7, 2, 4]}
    outputs = [((2, 0), 7), ((1, 0), 2), ((5, 0), 4)]
    task = task_module.Task(inputs_dict, outputs)
    self.assertEqual(checker.check_solution(task, solution_string), expected)

  @parameterized.named_parameters(
      ('good_solution', '((2 * a) + b)', True),
      ('ok_formatting', '2*a + b', True),
      ('ok_alternative', 'a + b + a', True),
      ('no_builtins', 'sum([a, a, b])', False),
      ('bad_syntax', '((2 * a) + b', False),
      ('wrong_result', 'a * a + b', False),
  )
  def test_check_solutions_arithmetic(self, solution_string, expected):
    inputs_dict = {'a': [2, 1, 5], 'b': [7, 2, 4]}
    outputs = [11, 4, 14]
    task = task_module.Task(inputs_dict, outputs)
    self.assertEqual(checker.check_solution(task, solution_string), expected)

  @parameterized.named_parameters(
      ('right_good', {'in1': ['abc', '12345', '?']}, ['bc', '45', '?'],
       'Right(in1, 2)', True),
      ('right_bad', {'in1': ['abc', '12345', '?']}, ['bc', '45', '?'],
       'Right(in1, 3)', False),
      ('concat_good', {'in1': ['abc', '12345', '?'], 'in2': ['XYZ', '0', '!!']},
       ['abcXYZ', '123450', '?!!'],
       'Concatenate(in1, in2)', True),
      ('concat_bad', {'in1': ['abc', '12345', '?'], 'in2': ['XYZ', '0', '!!']},
       ['abcXYZ', '123450', '?!!'],
       'Concatenate(in2, in1)', False),
      ('substitute_3_good', {'x': ['1?2?3', '???', '?']}, ['1!2!3', '!!!', '!'],
       'Substitute(x, "?", "!")', True),
      ('substitute_3_bad', {'x': ['1?2?3', '???', '?']}, ['1!2!3', '!!!', '!'],
       'Substitute(x, "?", "!!")', False),
      ('substitute_4_good', {'x': ['1?2?3', '???', '?']}, ['1?2!3', '?!?', '?'],
       'Substitute(x, "?", "!", 2)', True),
      ('substitute_4_bad', {'x': ['1?2?3', '???', '?']}, ['1?2!3', '?!?', '?'],
       'Substitute(x, "?", "!", 1)', False),

      ('path_depth_good', {'var_0': ['/this/is/a/path', '/home', '/a/b']},
       ['4', '1', '2'],
       'TO_TEXT(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, "/", ""))))', True),
      ('path_depth_bad', {'var_0': ['/this/is/a/path', '/home', '/a/b']},
       ['4', '11', '2'],
       'TO_TEXT(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, "/", ""))))', False),

      ('two_letter_acronym_capitalization_good',
       {'x': ['product area', 'Vice president']}, ['PA', 'VP'],
       'LEFT(REPLACE(UPPER(x), 2, MINUS(FIND(" ", x), 1), ""), 2)', True),
      ('two_letter_acronym_capitalization_bad',
       {'x': ['product area', 'Vice president']}, ['PA', 'VP'],
       'LEFT(REPLACE(x, 2, MINUS(FIND(" ", x), 1), ""), 2)', False),
  )
  def test_check_solutions_bustle(self, inputs_dict, outputs, solution_string,
                                  expected):
    task = task_module.Task(inputs_dict, outputs)
    self.assertEqual(checker.check_bustle_solution(task, solution_string),
                     expected)


if __name__ == '__main__':
  absltest.main()
