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

  def __init__(self, inputs_dict, outputs, solution=None):
    """inputs_dict: map from variable name to list of values (one list element per example)
    outputs: list of outputs, one for example
    solution: not sure? OperationValue"""
    self.inputs_dict = inputs_dict
    self.outputs = outputs
    self.solution = solution
    self.num_inputs = len(inputs_dict)
    self.num_examples = len(outputs)

  def __str__(self):
    return ('Task(\n'
            '    inputs_dict={},\n'
            '    outputs={},\n'
            '    solution={}\n'
            ')'.format(
                self.inputs_dict, self.outputs,
                self.solution.expression() if self.solution else None))
