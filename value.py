"""Defines the Value objects used in search."""

import abc


class Value(abc.ABC):
  """A value in search."""

  def __init__(self, value, weight):
    self.value = value
    self.weight = weight
    self.type = type(value)
    self._repr_cache = None

  def __repr__(self):
    """Returns a string representation of the value.

    Values are considered equal if and only if their string representations (as
    computed by this function) are equal.
    """
    if self._repr_cache is None:
      self._repr_cache = '{}:{!r}'.format(self.type.__name__, self.value)
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

  @abc.abstractmethod
  def expression(self):
    """Returns a code expression (as a string) that creates this value."""


class OperationValue(Value):
  """A Value resulting from the application of an Operation."""

  def __init__(self, value, weight, operation, arg_values):
    super(OperationValue, self).__init__(value, weight)
    self.operation = operation
    self.arg_values = arg_values

  def expression(self):
    """See base class."""
    return self.operation.expression(self.arg_values)


class ConstantValue(Value):
  """A constant value that is not created by any operation."""

  def expression(self):
    """See base class."""
    return repr(self.value)


class InputValue(Value):
  """A value provided by the user as an input."""

  def __init__(self, value, weight, name):
    """Initializes an InputValue to contain `value` with name `name`."""
    super(InputValue, self).__init__(value, weight)
    self.name = name

  def expression(self):
    """See base class."""
    return self.name


class OutputValue(Value):
  """A Value representing the user's desired output.

  This class is simply a wrapper aound the output value so that it can be
  compared to other Value objects.
  """

  def expression(self):
    """An OutputValue is not created from any expression."""
    raise NotImplementedError()
