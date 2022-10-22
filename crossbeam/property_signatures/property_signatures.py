# Copyright 2022 Google LLC
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

"""Property signatures for objects including lambdas.

Summary of approach:

type_property(x):
  given an object x, return its type as a one-hot list.
basic_properties(x):
  given an object x, return a list of boolean properties.
relevant(x):
  given an object x, return a list of related objects (including x) whose
  basic properties are relevant to understanding x.
basic_signature(x):
  type_property(x) + sum(basic_properties(r) for r in relevant(x))

compare_same_type(x, y):
  given two objects x and y of the same type, return a list of boolean
  properties representing the comparison.
compare(x, y):
  if type(x) == type(y):
    return compare_same_type(x, y)
  return (
      sum(compare_same_type(r, y) for r in relevant(x) if type(r) == type(y)) +
      sum(compare_same_type(x, r) for r in relevant(y) if type(x) == type(r)))

property_signature_single_example(inputs, output):
  basic_signature(output) + sum(basic_signature(i) + compare(i, output)
                                for i in inputs)

property_signature_single_object(x, output):
  basic_signature(x) + compare(x, output)

When dealing with multiple I/O examples, reduce across examples.

If desired, the property signatures can have fixed length, by adding a lot of
padding for the properties that don't apply (which would be omitted for variable
length signatures).
"""
# Booleans are a kind of int according to isinstance(). Use type() instead.
# pylint: disable=unidiomatic-typecheck

import itertools
import random
from typing import Any, List, Optional, Tuple, Type

from crossbeam.dsl import deepcoder_operations
from crossbeam.dsl import domains
from crossbeam.dsl import value as value_module

DEFAULT_VALUES = {
    bool: False,
    int: 0,
    list: [0],
}

# DeepCoder only uses lambdas that take int(s) as input.
VALUES_TO_TRY = {
    int: [-176, -67, -31, -7, -3, -2, -1, 0, 1, 2, 3, 5, 16, 25, 83, 130],
    #int: (
    #    [-176, -97, -55, -28, -13] +
    #    list(range(-10, 11)) +
    #    [16, 35, 66, 85, 143, 239]
    #),
}

# The maximum number of inputs for a lambda Value or an I/O example, if
# fixed-length signatures are desired. The inputs will be padded or truncated to
# match this number.
FIXED_NUM_IO_INPUTS = 3
FIXED_NUM_LAMBDA_INPUTS = 2


def _type_property(x) -> List[bool]:
  """Returns a one-hot List[bool] representation of type(x)."""
  type_x = type(x)
  is_lambda = getattr(x, 'num_free_variables', 0) > 0
  return [is_lambda, type_x is bool, type_x is int, type_x is list]


def _basic_properties(x) -> List[bool]:
  """Returns a list of basic properties for an object."""
  type_x = type(x)
  if type_x is bool:
    return [x]
  elif type_x is int:
    abs_x = abs(x)
    return [
        x == -1,
        x == 0,
        x == 1,
        x == 2,
        x > 0,
        x < 0,
        x % 2 == 0,
        abs_x < 5,
        abs_x < 10,
        abs_x < 15,
        abs_x < 20,
        abs_x < 30,
        abs_x < 40,
        abs_x < 50,
        abs_x < 75,
        abs_x < 100,
        abs_x < 150,
        abs_x < 200,
    ]
  elif type_x is list:
    sorted_x = list(sorted(x))
    reverse_sorted_x = list(reversed(sorted_x))
    num_unique = len(set(x))
    return [
        x == sorted_x,
        x == reverse_sorted_x,
        num_unique == 1,
        num_unique <= len(x) // 2,
        num_unique == len(x),
    ]
  else:
    raise NotImplementedError(f'x has unhandled type {type(x)}')


def _relevant(x) -> List[Any]:
  """Gets related objects relevant to understanding x."""
  type_x = type(x)
  if type_x is bool:
    return [x]
  elif type_x is int:
    return [x]
  elif type_x is list:
    orig_x = x
    len_x = len(x)
    if not x:
      x = DEFAULT_VALUES[type_x]
    max_x = max(x)
    min_x = min(x)
    return [orig_x, len_x, len(set(x)), max_x, min_x, max_x - min_x,
            sum(x), x[0], x[-1]]
  else:
    raise NotImplementedError(f'x has unhandled type {type(x)}')


def _basic_signature(x, fixed_length) -> List[Optional[bool]]:
  """Returns a signature representing a single concrete object."""
  if not fixed_length:
    return _type_property(x) + sum((_basic_properties(r) for r in _relevant(x)),
                                   [])
  else:
    type_x = type(x)
    basic_signature = _basic_signature(x, fixed_length=False)
    result = []
    for t in DEFAULT_VALUES:
      result.extend(basic_signature if type_x is t
                    else [None] * _BASIC_SIGNATURE_LENGTH_BY_TYPE[t])
    return result


_BASIC_SIGNATURE_LENGTH_BY_TYPE = {
    t: len(_basic_signature(DEFAULT_VALUES[t], fixed_length=False))
    for t in DEFAULT_VALUES
}


def _compare_same_type(x, y) -> List[bool]:
  """Returns a List[bool] comparing two objects of the same type."""
  type_x = type(x)
  assert type_x == type(y)
  if type_x is bool:
    return [x == y, x and y, x or y]
  elif type_x is int:
    abs_diff = abs(x - y)
    return [
        x == y,
        x < y,
        x > y,
        x != 0 and y % x == 0,
        y != 0 and x % y == 0,
        abs_diff < 2,
        abs_diff < 5,
        abs_diff < 10,
        abs_diff < 20,
        abs_diff < 50,
        abs_diff < 100,
        abs_diff < 200,
        (x >= 0) == (y >= 0),  # Signs are the same.
    ]
  elif type_x is list:
    unique_x = set(x)
    unique_y = set(y)
    len_x = len(x)
    len_y = len(y)
    return [
        x == y,
        len_x == len_y,
        len_x > len_y,
        len_x < len_y,
        abs(len_x - len_y) < 2,
        all(xi < yi for xi, yi in zip(x, y)),
        all(xi <= yi for xi, yi in zip(x, y)),
        all(xi > yi for xi, yi in zip(x, y)),
        all(xi >= yi for xi, yi in zip(x, y)),
        all(xi == yi for xi, yi in zip(x, y)),  # One is prefix of the other.
        unique_x == unique_y,
        unique_x.issubset(unique_y),
        unique_y.issubset(unique_x),
    ]
  else:
    raise NotImplementedError(f'x has unhandled type {type(x)}')


def _compare(x, y, fixed_length, x_types=None) -> List[Optional[bool]]:
  """Compares two concrete (non-lambda) objects of any type."""
  type_x = type(x)
  type_y = type(y)
  if not fixed_length:
    if type_x == type_y:
      return _compare_same_type(x, y)
    return (sum((_compare_same_type(r, y)
                 for r in _relevant(x) + _basic_properties(x)
                 if type(r) == type_y), []) +
            sum((_compare_same_type(x, r)
                 for r in _relevant(y) + _basic_properties(y)
                 if type_x == type(r)), []))
  else:
    result = []
    if x_types is None:
      type_pairs = itertools.product(DEFAULT_VALUES, repeat=2)
    else:
      type_pairs = itertools.product(x_types, DEFAULT_VALUES)
    for (type_1, type_2) in type_pairs:
      result.extend(_compare(x, y, fixed_length=False)
                    if (type_x, type_y) == (type_1, type_2)
                    else [None] * _COMPARE_LENGTH_BY_TYPES[(type_1, type_2)])
    return result


_COMPARE_LENGTH_BY_TYPES = {
    (type_1, type_2): len(_compare(DEFAULT_VALUES[type_1],
                                   DEFAULT_VALUES[type_2], fixed_length=False))
    for (type_1, type_2) in itertools.product(DEFAULT_VALUES, repeat=2)
}


def _property_signature_single_example(
    inputs: List[Any],
    output: Any,
    fixed_length: bool = True,
    fixed_num_inputs: int = FIXED_NUM_IO_INPUTS,
    input_types: Optional[List[Type[Any]]] = None) -> List[Optional[bool]]:
  """Returns a property signature for a single I/O example."""
  if not fixed_length:
    return _basic_signature(output, fixed_length) + sum(
        (_basic_signature(i, fixed_length) + _compare(i, output, fixed_length,
                                                      x_types=input_types)
         for i in inputs), [])
  else:
    if len(inputs) > fixed_num_inputs:
      inputs = inputs[:fixed_num_inputs]
    result = _basic_signature(output, fixed_length)
    basic_sig_length = len(result)
    result.extend(sum(
        (_basic_signature(i, fixed_length) + _compare(i, output, fixed_length,
                                                      x_types=input_types)
         for i in inputs), []))
    if len(inputs) < fixed_num_inputs:
      assert (len(result) - basic_sig_length) % len(inputs) == 0
      result.extend([None] * (
          (len(result) - basic_sig_length) // len(inputs) *  # Length per input
          (fixed_num_inputs - len(inputs))))  # Number of inputs to pad
    return  result


def _property_signature_single_object(
    x: Any,
    output: Any,
    fixed_length: bool = True) -> List[Optional[bool]]:
  """Returns a property signature for an object in context of an I/O example."""
  return _basic_signature(x, fixed_length) + _compare(x, output, fixed_length)


def _reduce_across_examples(
    signatures: List[List[Optional[bool]]]
) -> List[Tuple[float, bool, bool, float]]:
  """Reduce across examples (frac applicable, all True?, all False?, frac True)."""
  # All signatures should have the same length across examples.
  num_examples = len(signatures)
  assert num_examples
  signature_len = len(signatures[0])
  assert all(len(sig) == signature_len for sig in signatures)

  result = []
  for bools in zip(*signatures):
    bools_not_none = [x for x in bools if x is not None]
    num_not_none = len(bools_not_none)
    frac_true = sum(bools_not_none) / num_not_none if num_not_none else 0.5
    result.append((len(bools_not_none) / num_examples, frac_true))
  return result


def property_signature_io_examples(
    input_values: List[value_module.Value],
    output_value: value_module.Value,
    fixed_length: bool = True) -> List[Tuple[float, bool, bool, float]]:
  """Returns a property signature for a set of I/O examples."""
  num_examples = output_value.num_examples
  assert all(i_value.num_examples == num_examples for i_value in input_values)
  return _reduce_across_examples(
      [_property_signature_single_example(  # pylint: disable=g-complex-comprehension
          [i_value[example_index] for i_value in input_values],
          output_value[example_index],
          fixed_length=fixed_length,
          fixed_num_inputs=FIXED_NUM_IO_INPUTS)
       for example_index in range(num_examples)])

IO_EXAMPLES_SIGNATURE_LENGTH = len(property_signature_io_examples(
    [value_module.InputVariable([1, 2], 'in1')],
    value_module.OutputValue([-1, -2]),
    fixed_length=True))


def _property_signature_concrete_value(
    value: value_module.Value,
    output_value: value_module.Value,
    fixed_length: bool = True) -> List[Tuple[float, bool, bool, float]]:
  """Returns a property signature for a value w.r.t. a set of I/O examples."""
  assert value.num_free_variables == 0
  return _reduce_across_examples(
      [_property_signature_single_object(
          value[i], output_value[i], fixed_length=fixed_length)
       for i in range(output_value.num_examples)])


def run_lambda(
    value: value_module.Value,
) -> Optional[List[List[Tuple[List[Any], Any]]]]:
  """Runs a lambda on canonical values."""
  arity = value.num_free_variables
  assert arity > 0
  num_examples = value.num_examples
  io_pairs_per_example = [[] for _ in range(num_examples)]
  inputs_to_try = sum(VALUES_TO_TRY.values(), [])
  for i, (lambda_fn, io_pairs) in enumerate(zip(value.values, io_pairs_per_example)):
    shuffled_product = list(itertools.product(inputs_to_try, repeat=arity))
    random.Random(i).shuffle(shuffled_product)
    for inputs_list in shuffled_product:
      try:
        result = lambda_fn(*inputs_list)
        if domains.deepcoder_small_value_filter(result):
          io_pairs.append((inputs_list, result))
          if len(io_pairs) >= 10:
            break
      except:  # pylint: disable=bare-except
        pass
  if all(not pairs_for_example for pairs_for_example in io_pairs_per_example):
    # The lambda never ran successfully for any attempted input list for any I/O
    # example. Return None to signal this.
    return None
  return io_pairs_per_example


def is_value_valid(value: value_module.Value) -> bool:
  if value is None:
    return False
  if not value.num_free_variables:
    return True
  if not hasattr(value, 'lambda_exec_results'):
    value.lambda_exec_results = run_lambda(value)
  return value.lambda_exec_results != None


def _property_signature_lambda(
    value: value_module.Value,
    output_value: value_module.Value,
    fixed_length: bool = True) -> List[Tuple[float, bool, bool, float]]:
  """Returns a property signature for a lambda value."""
  if not hasattr(value, 'lambda_exec_results'):
    value.lambda_exec_results = run_lambda(value)
  io_pairs_per_example = value.lambda_exec_results
  if not io_pairs_per_example:
    # The lambda never ran successfully. We return all padding here, but such a
    # value shouldn't be kept in search.
    return [_REDUCED_PADDING] * _SIGNATURE_LENGTH_LAMBDA_VALUE
  signatures_to_reduce = []
  for example_index, pairs in enumerate(io_pairs_per_example):
    for inputs, output in pairs:
      signatures_to_reduce.append(
          _type_property(value) +
          _compare(output, output_value[example_index], fixed_length) +
          _property_signature_single_example(
            inputs, output, fixed_length,
            fixed_num_inputs=FIXED_NUM_LAMBDA_INPUTS,
            input_types=list(VALUES_TO_TRY)))
  return _reduce_across_examples(signatures_to_reduce)


def property_signature_value(
    value: value_module.Value,
    output_value: value_module.Value,
    fixed_length: bool = True) -> List[Tuple[float, bool, bool, float]]:
  """Returns a property signature for a Value w.r.t. to the output Value."""
  if not fixed_length:
    if value.num_free_variables:
      return _property_signature_lambda(value, output_value, fixed_length)
    else:
      return _property_signature_concrete_value(value, output_value,
                                                fixed_length)
  else:
    if value.num_free_variables:
      return ([_REDUCED_PADDING] * _SIGNATURE_LENGTH_CONCRETE_VALUE +
              _property_signature_lambda(value, output_value, fixed_length))
    else:
      return (_property_signature_concrete_value(value, output_value,
                                                 fixed_length) +
              [_REDUCED_PADDING] * _SIGNATURE_LENGTH_LAMBDA_VALUE)

_REDUCED_PADDING = (0.0, 0.5)
SIGNATURE_TUPLE_LENGTH = len(_REDUCED_PADDING)
_SIGNATURE_LENGTH_CONCRETE_VALUE = len(_property_signature_concrete_value(
    value_module.ConstantValue(0), value_module.OutputValue([1]),
    fixed_length=True))
_LAMBDA_VALUE = deepcoder_operations.Add().apply(
    [value_module.ConstantValue(10), value_module.get_free_variable(0)],
    free_variables=[value_module.get_free_variable(0)])
_SIGNATURE_LENGTH_LAMBDA_VALUE = len(_property_signature_lambda(
    _LAMBDA_VALUE, value_module.OutputValue([1]),
    fixed_length=True))

VALUE_SIGNATURE_LENGTH = len(property_signature_value(
    value_module.InputVariable([1, 2], 'in1'),
    value_module.OutputValue([-1, -2]),
    fixed_length=True))


def test():
  """Run some functions and print results to stdout."""
  import timeit  # pylint: disable=g-import-not-at-top
  fixed_length = True
  print(f'_BASIC_SIGNATURE_LENGTH_BY_TYPE: {_BASIC_SIGNATURE_LENGTH_BY_TYPE}')
  print(f'_COMPARE_LENGTH_BY_TYPES: {_COMPARE_LENGTH_BY_TYPES}')
  print(f'_SIGNATURE_LENGTH_CONCRETE_VALUE: {_SIGNATURE_LENGTH_CONCRETE_VALUE}')
  print(f'_SIGNATURE_LENGTH_LAMBDA_VALUE: {_SIGNATURE_LENGTH_LAMBDA_VALUE}')
  print()
  print(f'IO_EXAMPLES_SIGNATURE_LENGTH: {IO_EXAMPLES_SIGNATURE_LENGTH}')
  print(f'VALUE_SIGNATURE_LENGTH: {VALUE_SIGNATURE_LENGTH}')
  print()

  for x in [True, False, 0, 1, 15, [], [1, 2, 5], [-4, -4, -4]]:
    sig = _basic_signature(x, fixed_length)
    print(f'_basic_signature({x}) -> len {len(sig)}: {sig}')
  print()

  class Mock(object):

    def __getitem__(self, index):
      if self.num_examples == 1:
        index = 0
      return self.values[index]

  input_1 = Mock()
  input_1.values = [[1, 2, 4, 7], [0, 0], [5, -2, -20], [100]]
  input_1.num_examples = 4
  input_2 = Mock()
  input_2.values = [3, -6, 2, 50]
  input_2.num_examples = 4
  output = Mock()
  output.values = [[4, 5, 7, 10], [-6, -6], [7, 0, -18], [150]]
  output.num_examples = 4
  start_time = timeit.default_timer()
  sig = property_signature_io_examples([input_1, input_2], output, fixed_length)
  elapsed_time = timeit.default_timer() - start_time
  print(f'property_signature_io_examples(...) -> len {len(sig)}: {sig}')
  print(f'... took {elapsed_time} seconds')
  print()

  lambda_value = Mock()
  lambda_value.values = [
      lambda x, y: ([e + 3*y for e in x]  # pylint: disable=g-long-lambda
                    if type(x) == list and type(y) == int else None),
      lambda x, y: ([e + -1*y for e in x]  # pylint: disable=g-long-lambda
                    if type(x) == list and type(y) == int else None),
  ]
  lambda_value.num_examples = 2
  lambda_value.num_free_variables = 2

  start_time = timeit.default_timer()
  sig = property_signature_value(lambda_value, output, fixed_length)
  elapsed_time = timeit.default_timer() - start_time
  print(f'property_signature_value(lambda_value) -> len {len(sig)}')
  print(f'... took {elapsed_time} seconds')
  print()

  concrete_value = Mock()
  concrete_value.values = [123, 435, -12, 0]
  concrete_value.num_examples = 4
  concrete_value.num_free_variables = 0
  start_time = timeit.default_timer()
  sig = property_signature_value(concrete_value, output, fixed_length)
  elapsed_time = timeit.default_timer() - start_time
  print(f'property_signature_value(concrete_value) -> len {len(sig)}')
  print(f'... took {elapsed_time} seconds')


if __name__ == '__main__':
  test()

