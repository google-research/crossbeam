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

"""Checks whether string solutions adhere to the I/O examples."""


def check_solution(task, solution_string):
  """Checks whether solution_string passes all examples in the task."""
  for example_index in range(task.num_examples):
    namespace = {}
    for input_name, input_values in task.inputs_dict.items():
      namespace[input_name] = input_values[example_index]
    try:
      result = eval(solution_string, {'__builtins__': {}}, namespace)  # pylint: disable=eval-used
      if result != task.outputs[example_index]:
        return False
    except:  # pylint: disable=bare-except
      return False
  return True
