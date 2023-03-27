"""Computes the weight for string solutions.

A "long-form" solution reflects how a value is actually constructed, e.g.,

  ZipWith(lambda u1, u2: (lambda v1, v2: Subtract(v1, v2))(u2, u1), x, y)

has a lambda value `lambda v1, v2: Subtract(v1, v2)` as an argument to
ZipWith, and lambda values must be called with variable names, in this case
`(u2, u1)`.

The weight of this expression is easy to count: the `lambda ...:`
portions don't contribute any weight because they're basically baked into the
DSL, but otherwise every operation, variable name, and constant contributes 1
weight. The weight is 8: ZipWith, Subtract, v1, v2, u2, u1, x, y.

crossbeam.dsl.deepcoder_utils.simplify() simplifies this "long-form" solution
into a "short-form" solution:

  ZipWith(lambda u1, u2: Subtract(u2, u1), x, y)

This is just a syntactic change, so the short-form solution also has weight 8.
Basically, every operation inside a lambda (Subtract in this example)
contributes 1 plus the number of distinct `u` variables inside it, because
having N distinct `u` variable comes from calling a lambda (originally with `v`
variables) on an argument list of length N. Thus, we can equivalently count
weight 8 as: ZipWith, Subtract +2, u2, u1, x, y.

This may be applied multiple times:

  (long-form. weight 8: Map, Multiply, v1, Square, v1, v1, u1, x)
  Map(lambda u1: (lambda v1: Multiply(v1, (lambda v1: Square(v1))(v1)))(u1), x)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          vvvvvvvvvv
  Map(lambda u1: (lambda v1: Multiply(v1, Square(v1)))(u1), x)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 vvvvvvvvvvvvvvvvvvvvvvvv
  Map(lambda u1: Multiply(u1, Square(u1)), x)
  (short-form. weight 8: Map, Multiply +1, u1, Square +1, u1, x)

Extra care is needed to handle situations where a higher-order function uses a
lambda while inside a lambda itself.

This file implements counting the weight for long-form and short-form solutions.
"""

import re

from crossbeam.dsl import deepcoder_operations


def _end_index(tokens, start_index):
  """Finds corresponding , or ) denoting the end of this arg in an arg list."""
  paren_count = 0
  i = start_index
  found_open_paren = False
  while i < len(tokens):
    if tokens[i] in (',', ')') and found_open_paren and paren_count == 0:
      return i
    elif tokens[i] == '(':
      paren_count += 1
      found_open_paren = True
    elif tokens[i] == ')':
      paren_count -= 1
    i += 1
  return i


def solution_weight(solution: str) -> int:
  """Counts the weight of a solution."""
  # This implementation assumes the solution is either short-form (without `v1`
  # or `v2` variables) or long-form (with those variables), and syntactically
  # correct. It might not detect cases where the solution has syntax errors.
  long_form = re.search(r'v\d', solution)

  operation_names = {op.name for op in deepcoder_operations.get_operations()}
  for c in ['(', ')', ',', ':']:
    solution = solution.replace(c, f' {c} ')
  tokens = solution.split()
  weight = 0

  i = 0
  while i < len(tokens):
    token = tokens[i]

    if token == 'lambda':
      # Skip variables until the next `:` token.
      while tokens[i] != ':':
        i += 1
      if not long_form:
        # Add bonus weight for operations inside this lambda (excluding inner
        # lambdas).
        lambda_end_index = _end_index(tokens, i)
        op_index = i + 1
        while op_index < lambda_end_index:
          if tokens[op_index] == 'lambda':
            # Jump past this inner lambda.
            op_index = _end_index(tokens, op_index)
          elif tokens[op_index] in operation_names:
            # Count the number of distinct `u` variables inside this operation.
            distinct_vars = set()
            inner_index = op_index + 1
            op_end_index = _end_index(tokens, op_index)
            while inner_index < op_end_index:
              if tokens[inner_index] == 'lambda':
                # Jump past this inner lambda.
                inner_index = _end_index(tokens, inner_index)
              elif re.fullmatch(r'u\d', tokens[inner_index]):
                distinct_vars.add(tokens[inner_index])
              inner_index += 1
            weight += len(distinct_vars)
          op_index += 1

    elif token in ['(', ')', ',']:
      # Ignore these syntactic tokens that don't contribute weight.
      pass
    elif (
        # An integer constant contributes weight 1.
        re.fullmatch(r'-?\d', token) or
        # An operation counts 1, and any additional weight was already covered
        # in the lambda case.
        token in operation_names or
        # A variable like `u1`, used as bound variables in lambdas.
        re.fullmatch(r'u\d', token) or
        # A single-letter variable like `x`, used as inputs in handwritten
        # tasks.
        re.fullmatch(r'[a-z]', token) or
        # A variable like `x1`, used as inputs in synthetic tasks.
        re.fullmatch(r'x\d', token)
    ):
      weight += 1
    elif long_form and re.fullmatch(r'v\d', token):
      # A variable like `v1`, used as free variables in long-form lambdas.
      weight += 1
    else:
      raise ValueError(f'Unhandled token `{token}` at position {i}: {tokens}')

    i += 1

  return weight
