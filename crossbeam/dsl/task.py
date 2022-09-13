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

"""Tasks (input-output examples) for synthesis."""


class Task(object):
  """Stores multiple I/O examples for a single task."""

  def __init__(self, inputs_dict, outputs, solution=None, name=None, **kwargs):
    """Task constructor.

    Args:
      inputs_dict: map from variable name to list of values (one list element
          per example)
      outputs: list of outputs, one for example
      solution: OperationValue or string
      name: optional name for the task
      **kwargs: any other arguments will get stored in the `kwargs` field
    """
    self.inputs_dict = inputs_dict
    self.outputs = outputs
    self.solution = solution
    self.name = name
    self.num_inputs = len(inputs_dict)
    self.num_examples = len(outputs)
    self.kwargs = kwargs

  def __str__(self):
    solution_str = (self.solution if isinstance(self.solution, str)
                    else self.solution.expression() if self.solution
                    else None)
    return (f'Task(\n'
            f'    name={self.name!r},\n'
            f'    inputs_dict={self.inputs_dict},\n'
            f'    outputs={self.outputs},\n'
            f'    solution={solution_str!r}\n'
            f')')
