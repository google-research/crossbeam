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

"""Operations for the pair creation DSL."""

from crossbeam.dsl import operation_base


class PairOperation(operation_base.OperationBase):
  """An operation that creates a pair."""

  def __init__(self):
    super(PairOperation, self).__init__('Pair', 2)

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return (left, right)

  def tokenized_expression(self, arg_values):
    """See base class."""
    left, right = arg_values
    return (['('] + left.tokenized_expression() + [', ']
            + right.tokenized_expression() + [')'])


class TripleOperation(operation_base.OperationBase):
  """An operation that creates a triple."""

  def __init__(self):
    super(TripleOperation, self).__init__('Triple', 3)

  def apply_single(self, raw_args):
    """See base class."""
    left, mid, right = raw_args
    return (left, mid, right)

  def tokenized_expression(self, arg_values):
    """See base class."""
    left, mid, right = arg_values
    return (['('] + left.tokenized_expression() + [', ']
            + mid.tokenized_expression() + [', ']
            + right.tokenized_expression() + [')'])


def get_operations():
  return [
      PairOperation(),
      TripleOperation(),
  ]
