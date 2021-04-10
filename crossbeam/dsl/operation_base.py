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

"""Defines the Operations used in search."""

import abc

from crossbeam.dsl import value as value_module


class OperationBase(abc.ABC):
  """An operation used in synthesis."""

  def __init__(self, name, arity, weight=1):
    self.name = name
    self.arity = arity
    self.weight = weight

  def apply(self, arg_values):
    """Applies the operation to a list of arguments, for all examples."""
    num_examples = arg_values[0].num_examples
    try:
      results = [self.apply_single([arg_value[i] for arg_value in arg_values])
                 for i in range(num_examples)]
    except Exception:  # pylint: disable=broad-except
      # Some exception occured in apply_single. This is ok, just throw out this
      # value.
      return None
    return value_module.OperationValue(results, self, arg_values)

  @abc.abstractmethod
  def apply_single(self, raw_arg_values):
    """Applies the operation to a list of arguments, for 1 example."""

  @abc.abstractmethod
  def expression(self, arg_values):
    """Returns a code expression for an application of this operation."""
