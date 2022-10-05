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
import functools

_FUNCTION_TYPE = type(lambda x: x)  # pylint: disable=invalid-name


class Value(abc.ABC):
  """Values in search, one for each I/O example."""

  def __init__(self, values, free_variables=None):
    # `values` may be concrete execution results on the inputs, or function
    # objects. In either case, there should be 1 such value for each example.
    assert values
    self.values = values
    self.type = type(values[0])  # Could be the function type.
    self.num_examples = len(values)
    self.free_variables = [] if free_variables is None else free_variables
    self.num_free_variables = len(free_variables) if free_variables else 0
    self.contains_lambda = False  # Will be updated in OperationValue.
    self._repr_cache = None

  def __repr__(self):
    """Returns a string representation of the value.

    Values are considered equal if and only if their string representations (as
    computed by this function) are equal.

    If there are no free variables, use the actual values (execution results).
    If there are free variables, use the code expression.
    """
    if self._repr_cache is None:
      if isinstance(self, (FreeVariable, BoundVariable)) or self.free_variables:
        self._repr_cache = self.expression()
      else:
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
    return (self.num_free_variables == other.num_free_variables and
            repr(self) == repr(other))

  def __ne__(self, other):
    """Returns whether this Value object is not equal to `other`."""
    return not self == other

  def __getitem__(self, index):
    """Gets the raw value for the example with the specified index."""
    if self.num_examples == 1:
      # Some values (Constants, FreeVariables, BoundVariables) have only 1 value
      # even though there might be multiple examples. Return the same value for
      # each example.
      index = 0
    return self.values[index]

  def expression(self):
    """Returns a code expression (as a string) that creates this value."""
    return ''.join(self.tokenized_expression())

  @abc.abstractmethod
  def tokenized_expression(self):
    """Returns a code expression (tokenized) that creates this value."""

  @abc.abstractmethod
  def get_weight(self):
    """Returns this expression's weight, computed recursively."""

  def __getstate__(self):
    state = self.__dict__.copy()
    if self.type == _FUNCTION_TYPE:
      # We cannot pickle lambdas or the function type. Skip them.
      state['values'] = []
      state['type'] = 'FUNCTION'
    return state


class Variable(Value):
  """A variable name."""

  def __init__(self, name, values=None, weight=1, is_free=False):
    if values is None:
      values = [f'{self.__class__.__name__}_{name}']
    super(Variable, self).__init__(values,
                                   free_variables=[self] if is_free else [])
    self.name = name
    self._weight = weight

  def tokenized_expression(self):
    return [self.name]

  def get_weight(self):
    return self._weight


class FreeVariable(Variable):
  """A free variable name, which implies the outermost Value is a lambda."""

  def __init__(self, name):
    super(FreeVariable, self).__init__(name, is_free=True)


@functools.lru_cache(maxsize=None)
def get_free_variable(i):
  return FreeVariable(f'v{i + 1}')


class BoundVariable(Variable):
  """A bound variable name, bound to a lambda in a lambda-requiring function."""

  def __init__(self, name):
    super(BoundVariable, self).__init__(name)


@functools.lru_cache(maxsize=None)
def get_bound_variable(i):
  return BoundVariable(f'u{i + 1}')


class InputVariable(Variable):
  """A value provided by the user as an input variable."""

  def __init__(self, values, name):
    super(InputVariable, self).__init__(name, values=values)


class OperationValue(Value):
  """A Value resulting from the application of an Operation."""

  def __init__(self, values, operation, arg_values, arg_variables=None,
               free_variables=None):
    super(OperationValue, self).__init__(values, free_variables)
    self.operation = operation
    self.arg_values = arg_values
    self.arg_variables = ([()] * operation.arity if arg_variables is None
                          else arg_variables)
    self.contains_lambda = (bool(free_variables) or
                            any(v.contains_lambda for v in arg_values))

  def tokenized_expression(self):
    return self.operation.tokenized_expression(
        self.arg_values, self.arg_variables, self.free_variables)

  def get_weight(self):
    return (self.operation.weight + sum(v.get_weight() for v in self.arg_values)
            + self.num_free_variables)


class ConstantValue(Value):
  """A constant value that is not created by any operation."""

  def __init__(self, constant, weight=1):
    super(ConstantValue, self).__init__([constant])
    self.constant = constant
    self._weight = weight

  def tokenized_expression(self):
    return [repr(self.constant)]

  def get_weight(self):
    return self._weight


class OutputValue(Value):
  """A Value representing the user's desired output.

  This class is simply a wrapper aound the output values so that it can be
  compared to other Value objects.
  """

  def tokenized_expression(self):
    """An OutputValue is not created from any expression."""
    raise NotImplementedError()

  def get_weight(self):
    raise NotImplementedError()
