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

import collections
import functools

from crossbeam.dsl import bustle_operations


def check_solution(task, solution_string, ops_namespace=None):
  """Checks whether solution_string passes all examples in the task."""
  for example_index in range(task.num_examples):
    namespace = ops_namespace.copy() if ops_namespace else {}
    for input_name, input_values in task.inputs_dict.items():
      namespace[input_name] = input_values[example_index]
    try:
      result = eval(solution_string, {'__builtins__': {}}, namespace)  # pylint: disable=eval-used
      if result != task.outputs[example_index]:
        return False
    except:  # pylint: disable=bare-except
      return False
  return True


def _bustle_ops_namespace():
  """Constructs a namespace that supports calling BUSTLE operations."""
  name_to_ops = collections.defaultdict(list)
  for op in bustle_operations.get_operations():
    name_to_ops[op.name].append(op)
  ops_namespace = {}
  for name, ops_with_name in name_to_ops.items():
    def run_op(name, ops_with_name, *args):
      for op in ops_with_name:
        if len(args) == op.arity:
          return op.apply_single(args)
      raise ValueError('Op {} got bad arity {}'.format(name, len(args)))
    partial_run_op = functools.partial(run_op, name, ops_with_name)
    ops_namespace[name] = partial_run_op
    ops_namespace[name.upper()] = partial_run_op
  return ops_namespace


BUSTLE_OPS_NAMESPACE = _bustle_ops_namespace()


def check_bustle_solution(task, solution_string):
  """Checks whether solution_string passes all examples in the task."""
  return check_solution(task, solution_string, BUSTLE_OPS_NAMESPACE)
