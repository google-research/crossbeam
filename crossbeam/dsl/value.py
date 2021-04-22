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

"""Defines the Value objects used in search."""

import abc


class Value(abc.ABC):
  """Values in search, one for each I/O example."""

  def __init__(self, values, weight):
    assert isinstance(values, (list, tuple)) and values
    self.values = values
    self.weight = weight
    self.type = type(values[0])
    self.num_examples = len(values)
    self._repr_cache = None

  def __repr__(self):
    """Returns a string representation of the value.

    Values are considered equal if and only if their string representations (as
    computed by this function) are equal.
    """
    if self._repr_cache is None:
      self._repr_cache = '[' + ', '.join('{}:{!r}'.format(type(v).__name__, v)
                                         for v in self.values) + ']'
    return self._repr_cache

  def __hash__(self):
    """Implements hash so that Value objects can be used as dict keys."""
    return hash(repr(self))

  def __eq__(self, other):
    """Returns whether this Value object is equal to `other`.

    Args:
      other: The other object to compare to.

    Values are considered equal if and only if their string representations (as
    computed by __repr__) are equal.
    """
    if not isinstance(other, Value):
      return NotImplemented
    return repr(self) == repr(other)

  def __ne__(self, other):
    """Returns whether this Value object is not equal to `other`."""
    return not self == other

  def __getitem__(self, index):
    """Gets the raw value for the example with the specified index."""
    return self.values[index]

  def expression(self):
    """Returns a code expression (as a string) that creates this value."""
    return ''.join(self.tokenized_expression())

  @abc.abstractmethod
  def tokenized_expression(self):
    """Returns a code expression (tokenized) that creates this value."""


class OperationValue(Value):
  """A Value resulting from the application of an Operation."""

  def __init__(self, value, operation, arg_values):
    super(OperationValue, self).__init__(
        value, weight=operation.weight + sum(a.weight for a in arg_values))
    self.operation = operation
    self.arg_values = arg_values

  def tokenized_expression(self):
    """See base class."""
    return self.operation.tokenized_expression(self.arg_values)


class ConstantValue(Value):
  """A constant value that is not created by any operation."""

  def __init__(self, constant, num_examples, weight=1):
    super(ConstantValue, self).__init__([constant] * num_examples, weight)
    self.constant = constant

  def tokenized_expression(self):
    """See base class."""
    return [repr(self.constant)]


class InputValue(Value):
  """A value provided by the user as an input."""

  def __init__(self, values, name, weight=1):
    """Initializes an InputValue to contain `values` with name `name`."""
    super(InputValue, self).__init__(values, weight)
    self.name = name

  def tokenized_expression(self):
    """See base class."""
    return [self.name]


class OutputValue(Value):
  """A Value representing the user's desired output.

  This class is simply a wrapper aound the output values so that it can be
  compared to other Value objects.
  """

  def __init__(self, values, weight=-1):
    """Initializes an OutputValue with a sentinel weight."""
    super(OutputValue, self).__init__(values, weight)

  def tokenized_expression(self):
    """An OutputValue is not created from any expression."""
    raise NotImplementedError()
