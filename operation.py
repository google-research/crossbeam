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
#
"""Defines the Operations used in search."""

import abc


class Operation(abc.ABC):
  """An operation used in synthesis."""

  def __init__(self, name, arity):
    self.name = name
    self.arity = arity

  @abc.abstractmethod
  def apply(self, arg_values):
    """Applies the operation to a list of arguments."""

  @abc.abstractmethod
  def expression(self, arg_values):
    """Returns a code expression for an application of this operation."""


class PairOperation(Operation):
  """An operation that creates a pair."""

  def __init__(self):
    super(PairOperation, self).__init__('Pair', 2)

  def apply(self, arg_values):
    """See base class."""
    left, right = arg_values
    return (left, right)

  def expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return '({}, {})'.format(left.expression(), right.expression())


OPERATIONS = [
    PairOperation(),
]
