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

  def __hash__(self):
    return hash(repr(self))

  def __eq__(self, other):
    return repr(self) == repr(other)

  def __repr__(self):
    return type(self).__name__

  def arg_types(self):
    """The types of this operation's arguments, or None to allow any types."""
    return None

  def apply(self, arg_values):
    """Applies the operation to a list of arguments, for all examples."""
    num_examples = arg_values[0].num_examples
    arg_types = self.arg_types()  # pylint: disable=assignment-from-none
    if arg_types is not None and arg_types != tuple(x.type for x in arg_values):
      return None
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

  def expression(self, arg_values):
    """Returns a code expression for an application of this operation."""
    return ''.join(self.tokenized_expression(arg_values))

  @abc.abstractmethod
  def tokenized_expression(self, arg_values):
    """Returns a tokenized expression for an application of this operation."""
