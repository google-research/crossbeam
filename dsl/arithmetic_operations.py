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

"""Operations for the integer arithmetic DSL."""

from crossbeam.dsl import operation_base


class AddOperation(operation_base.OperationBase):
  """An operation that adds 2 numbers."""

  def __init__(self):
    super(AddOperation, self).__init__('AddOperation', 2)

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left + right

  def expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return '({} + {})'.format(left.expression(), right.expression())


class SubtractOperation(operation_base.OperationBase):
  """An operation that subtracts 2 numbers."""

  def __init__(self):
    super(SubtractOperation, self).__init__('SubtractOperation', 2)

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left - right

  def expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return '({} - {})'.format(left.expression(), right.expression())


class MultiplyOperation(operation_base.OperationBase):
  """An operation that multiplies 2 numbers."""

  def __init__(self):
    super(MultiplyOperation, self).__init__('MultiplyOperation', 2)

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left * right

  def expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return '({} * {})'.format(left.expression(), right.expression())


class IntDivideOperation(operation_base.OperationBase):
  """An operation that divides 2 integers."""

  def __init__(self):
    super(IntDivideOperation, self).__init__('IntDivideOperation', 2)

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left // right

  def expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return '({} // {})'.format(left.expression(), right.expression())


def get_operations():
  return [
      AddOperation(),
      SubtractOperation(),
      MultiplyOperation(),
      IntDivideOperation(),
  ]
