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

"""Operations for the BUSTLE string manipulation domain."""

import re

from crossbeam.dsl import operation_base


class BustleOperation(operation_base.OperationBase):
  """All Bustle operations have the same form of tokenized expression."""

  def tokenized_expression(self, arg_values, arg_variables, free_variables):
    del arg_variables, free_variables
    tokens = [self.name, '(']
    for i, arg in enumerate(arg_values):
      if i > 0:
        tokens.append(', ')
      tokens.extend(arg.tokenized_expression())
    tokens.append(')')
    return tokens


class Add(BustleOperation):
  """Adds two ints."""

  def __init__(self):
    super(Add, self).__init__('Add', 2)

  def arg_types(self):
    """See base class."""
    return int, int

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left + right


class Concatenate(BustleOperation):
  """Concatenates 2 strings."""

  def __init__(self):
    super(Concatenate, self).__init__('Concatenate', 2)

  def arg_types(self):
    """See base class."""
    return str, str

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left + right


class Find2(BustleOperation):
  """Finds index of first string within second string."""

  def __init__(self):
    super(Find2, self).__init__('Find', 2)

  def arg_types(self):
    """See base class."""
    return str, str

  def apply_single(self, raw_args):
    """See base class."""
    find, within = raw_args
    index = within.find(find)
    if index < 0:
      raise ValueError()
    return index + 1


class Find3(BustleOperation):
  """Finds index of first string within second string, starting at index."""

  def __init__(self):
    super(Find3, self).__init__('Find', 3)

  def arg_types(self):
    """See base class."""
    return str, str, int

  def apply_single(self, raw_args):
    """See base class."""
    find, within, start = raw_args
    if start <= 0 or start > len(within):
      raise ValueError()
    index = within.find(find, start - 1)
    if index < 0:
      raise ValueError()
    return index + 1


class Left(BustleOperation):
  """Gets prefix of string."""

  def __init__(self):
    super(Left, self).__init__('Left', 2)

  def arg_types(self):
    """See base class."""
    return str, int

  def apply_single(self, raw_args):
    """See base class."""
    txt, num_chars = raw_args
    if num_chars < 0:
      raise ValueError()
    return txt[:num_chars]


class Len(BustleOperation):
  """Gets length of string."""

  def __init__(self):
    super(Len, self).__init__('Len', 1)

  def arg_types(self):
    """See base class."""
    return (str,)

  def apply_single(self, raw_args):
    """See base class."""
    return len(raw_args[0])


class Mid(BustleOperation):
  """Gets a substring of a string, given start position and length."""

  def __init__(self):
    super(Mid, self).__init__('Mid', 3)

  def arg_types(self):
    """See base class."""
    return str, int, int

  def apply_single(self, raw_args):
    """See base class."""
    txt, start, num_chars = raw_args
    if start <= 0 or num_chars < 0:
      raise ValueError()
    return txt[start - 1 : start - 1 + num_chars]


class Minus(BustleOperation):
  """Subtracts two ints."""

  def __init__(self):
    super(Minus, self).__init__('Minus', 2)

  def arg_types(self):
    """See base class."""
    return int, int

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left - right


class Replace(BustleOperation):
  """Performs string replacement."""

  def __init__(self):
    super(Replace, self).__init__('Replace', 4)

  def arg_types(self):
    """See base class."""
    return str, int, int, str

  def apply_single(self, raw_args):
    """See base class."""
    text, start, num_chars, replacement = raw_args
    if start <= 0 or num_chars < 0:
      raise ValueError()
    return text[:start - 1] + replacement + text[start - 1 + num_chars:]


class Right(BustleOperation):
  """Gets suffix of string."""

  def __init__(self):
    super(Right, self).__init__('Right', 2)

  def arg_types(self):
    """See base class."""
    return str, int

  def apply_single(self, raw_args):
    """See base class."""
    txt, num_chars = raw_args
    if num_chars < 0:
      raise ValueError()
    return txt[max(0, len(txt) - num_chars):]


class Trim(BustleOperation):
  """Removes leading, trailing, and repeated spaces in a string."""

  def __init__(self):
    super(Trim, self).__init__('Trim', 1)

  def arg_types(self):
    """See base class."""
    return (str,)

  def apply_single(self, raw_args):
    """See base class."""
    txt = raw_args[0]
    return re.sub(' +', ' ', txt.strip(' '))


class Lower(BustleOperation):
  """Lowercases a string."""

  def __init__(self):
    super(Lower, self).__init__('Lower', 1)

  def arg_types(self):
    """See base class."""
    return (str,)

  def apply_single(self, raw_args):
    """See base class."""
    return raw_args[0].lower()


class Upper(BustleOperation):
  """Uppercases a string."""

  def __init__(self):
    super(Upper, self).__init__('Upper', 1)

  def arg_types(self):
    """See base class."""
    return (str,)

  def apply_single(self, raw_args):
    """See base class."""
    return raw_args[0].upper()


class Proper(BustleOperation):
  """Propercases (title cases) a string."""

  def __init__(self):
    super(Proper, self).__init__('Proper', 1)

  def arg_types(self):
    """See base class."""
    return (str,)

  def apply_single(self, raw_args):
    """See base class."""
    return raw_args[0].title()


class Rept(BustleOperation):
  """Repeats a string."""

  def __init__(self):
    super(Rept, self).__init__('Rept', 2)

  def arg_types(self):
    """See base class."""
    return str, int

  def apply_single(self, raw_args):
    """See base class."""
    txt, times = raw_args
    if times < 0:
      raise ValueError()
    if len(txt) * times > 100:
      raise ValueError()
    return txt * times


class Substitute3(BustleOperation):
  """Performs string substitution."""

  def __init__(self):
    super(Substitute3, self).__init__('Substitute', 3)

  def arg_types(self):
    """See base class."""
    return str, str, str

  def apply_single(self, raw_args):
    """See base class."""
    source, search, replace = raw_args
    if search == '':  # pylint: disable=g-explicit-bool-comparison
      return source
    return source.replace(search, replace)


class Substitute4(BustleOperation):
  """Performs string substitution for a particular occurrence."""

  def __init__(self):
    super(Substitute4, self).__init__('Substitute', 4)

  def arg_types(self):
    """See base class."""
    return str, str, str, int

  def apply_single(self, raw_args):
    """See base class."""
    source, search, replace, occurrence = raw_args
    if occurrence < 0:
      raise ValueError()
    if search == '':  # pylint: disable=g-explicit-bool-comparison
      return source
    if occurrence == 0:
      return source.replace(search, replace)

    index = -1
    for _ in range(occurrence):
      index += 1
      index = source.find(search, index)
      if index == -1:
        break
    if index != -1:
      return source[:index] + replace + source[index + len(search):]
    else:
      return source


class ToText(BustleOperation):
  """Converts an integer to a string."""

  def __init__(self):
    super(ToText, self).__init__('To_Text', 1)

  def arg_types(self):
    """See base class."""
    return (int,)

  def apply_single(self, raw_args):
    """See base class."""
    return str(raw_args[0])


class If(BustleOperation):
  """Performs if-then-else."""

  def __init__(self):
    super(If, self).__init__('If', 3)

  def arg_types(self):
    """See base class."""
    return bool, str, str

  def apply_single(self, raw_args):
    """See base class."""
    condition, true_result, false_result = raw_args
    return true_result if condition else false_result


class Exact(BustleOperation):
  """Checks if two strings exactly match."""

  def __init__(self):
    super(Exact, self).__init__('Exact', 2)

  def arg_types(self):
    """See base class."""
    return str, str

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left == right


class Gt(BustleOperation):
  """Checks if the first int is greater than the second int."""

  def __init__(self):
    super(Gt, self).__init__('Gt', 2)

  def arg_types(self):
    """See base class."""
    return int, int

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left > right


class Gte(BustleOperation):
  """Checks if the first int is greater than or equal to the second int."""

  def __init__(self):
    super(Gte, self).__init__('Gte', 2)

  def arg_types(self):
    """See base class."""
    return int, int

  def apply_single(self, raw_args):
    """See base class."""
    left, right = raw_args
    return left >= right


def get_operations():
  return [
      Add(),
      Concatenate(),
      Find2(),
      Find3(),
      Left(),
      Len(),
      Mid(),
      Minus(),
      Replace(),
      Right(),
      Trim(),
      Lower(),
      Upper(),
      Proper(),
      Rept(),
      Substitute3(),
      Substitute4(),
      ToText(),
      If(),
      Exact(),
      Gt(),
      Gte(),
  ]


def bustle_op_names():
  return [op.name for op in get_operations()]
