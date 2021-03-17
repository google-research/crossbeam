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
