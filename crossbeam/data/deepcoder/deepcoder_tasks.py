"""Handwritten DeepCoder-style tasks using lambdas."""

import collections
import functools
import re

from absl import app

from crossbeam.dsl import deepcoder_operations

_OPS_NAMESPACE = {
    op.name: functools.partial(lambda *args, op: op.apply_single(args), op=op)
    for op in deepcoder_operations.get_operations()
}

Task = collections.namedtuple(
    'Task', ['name', 'inputs_dict', 'output', 'solution'])

HANDWRITTEN_TASKS = [
    # Tasks without higher-order functions.
    # TODO(kshi)

    # Tasks primarily using Map.
    ############################
    Task(
        name='cube',
        inputs_dict={
            'x': [[3], [4, 1, 2], [-1, 5, 0, -4, 2, 3, -2]],
        },
        output=[[27], [64, 1, 8], [-1, 125, 0, -64, 8, 27, -8]],
        solution='Map(lambda u1: Multiply(u1, Square(u1)), x)',
    ),
    Task(
        name='sort_square',
        inputs_dict={
            'x': [[3, 4, 5], [2, 6, -3], [3, 1, -1, 6, 0, 3, 2, 7, -5]],
        },
        output=[[9, 16, 25], [4, 9, 36], [0, 1, 1, 4, 9, 9, 25, 36, 49]],
        solution='Sort(Map(lambda u1: Square(u1), x))',
    ),
    Task(
        name='triangular_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                  [0, 5, 2, 1, 6],
                  [7, 2, 5, 9, 2, 0, 4, 1, 3]],
        },
        output=[[1, 3, 6, 10],
                [0, 15, 3, 1, 21],
                [28, 3, 15, 45, 3, 0, 10, 1, 6]],
        solution='Map(lambda u1: IntDivide(Multiply(u1, Add(u1, 1)), 2), x)',
    ),
    Task(
        name='shift_min_to_zero',
        inputs_dict={
            'x': [[3, 4, 7, 9],
                  [16, 10, 25, 16, 83],
                  [-4, 3, 1, 0, -5, 5]],
        },
        output=[[0, 1, 4, 6],
                [6, 0, 15, 6, 73],
                [1, 8, 6, 5, 0, 10]],
        solution='Map(lambda u1: Subtract(u1, Minimum(x)), x)',
    ),
    Task(
        name='shift_element_to_zero',
        inputs_dict={
            'x': [[3, 4, 7, 9],
                  [16, 10, 25, 16, 83],
                  [-4, 3, 1, 0, -5, 5]],
            'i': [1, 3, 2],
        },
        output=[[-1, 0, 3, 5],
                [0, -6, 9, 0, 67],
                [-5, 2, 0, -1, -6, 4]],
        solution='Map(lambda u1: Subtract(u1, Access(i, x)), x)',
    ),
    Task(
        name='gather',
        inputs_dict={
            'x': [[4, 2, 6, 8],
                  [11, 3, 20, -5, 7],
                  [3, -6, 4, -5, 3, 9, -2, 0, 1, -4]],
            'i': [[0, 2, 1, 3],
                  [1, 4, 2, 1, 3, 1, 4, 1],
                  [2, 0, 2, 8, 4, 2, 3, 5]],
        },
        output=[[4, 6, 2, 8],
                [3, 7, 20, 3, -5, 3, 7, 3],
                [4, 3, 4, 1, 3, 4, -5, 9]],
        solution='Map(lambda u1: Access(u1, x), i)',
    ),

    # Tasks primarily using Filter.
    ###############################
    Task(
        name='filter_greater',
        inputs_dict={
            'data': [[1, 3, 4, 2],
                     [6, 4, 3, 5, 9, 2],
                     [25, 0, 79, -1, -45, 31, -4, 7, -2, 11]],
            'limit': [2, 4, -3],
        },
        output=[[3, 4], [6, 5, 9], [25, 0, 79, -1, 31, 7, -2, 11]],
        solution='Filter(lambda u1: Greater(u1, limit), data)',
    ),
    # Tasks primarily using Count.
    ##############################
    Task(
        name='count_element',
        inputs_dict={
            'data': [[5, 7, 8, 7, 9, 6],
                     [5, 7, 8, 7, 9, 6],
                     [7, 0, 7, 0, 7, 7],
                     [42, 34, 42, 42, 42, 56, 42, 38, 42, 42],
                     [42, 34, 42, 42, 38, 56, 42, 38, 42, 42]],
            'query': [7, 8, 7, 42, 42],
        },
        output=[2, 1, 4, 7, 6],
        solution='Count(lambda u1: Equal(u1, query), data)',
    ),

    # Tasks primarily using ZipWith.
    ################################
    Task(
        name='dot_product',
        inputs_dict={
            'x1': [[2, 0], [3, 1, 2], [7, 8, 1, 4]],
            'x2': [[10, 5], [8, 2, -5], [1, 0, 4, 2]],
        },
        output=[20, 16, 19],
        solution='Sum(ZipWith(lambda u1, u2: Multiply(u1, u2), x1, x2))',
    ),
    Task(
        name='elementwise_mean',
        inputs_dict={
            'x1': [[8, 0], [12, 3, 7], [-5, 4, 3, -7, 11, 0]],
            'x2': [[2, 4], [4, 1, 7], [3, 6, -3, -11, 19, -6]],
        },
        output=[[5, 2], [8, 2, 7], [-1, 5, 0, -9, 15, -3]],
        solution='ZipWith(lambda u1, u2: IntDivide(Add(u1, u2), 2), x1, x2)',
    ),
    Task(
        name='three_way_sum',
        inputs_dict={
            'x1': [[2], [6, 3], [4, 2, 1]],
            'x2': [[6], [2, 7], [8, 4, 0]],
            'x3': [[3], [0, 0], [0, 1, 8]],
        },
        output=[[11], [8, 10], [12, 7, 9]],
        solution=('ZipWith(lambda u1, u2: Add(u1, u2), x1, '
                  'ZipWith(lambda u1, u2: Add(u1, u2), x2, x3))'),
    ),

    # Tasks primarily using Scanl1.
    ###############################
    Task(
        name='running_max',
        inputs_dict={
            'x': [[1, 3, 6, 20],
                  [4, 2, 6, 3, 1, 7, 3, 9],
                  [-6, -5, -2, 0, -1, 3, 3, 2, 5, 4]],
        },
        output=[[1, 3, 6, 20],
                [4, 4, 6, 6, 6, 7, 7, 9],
                [-6, -5, -2, 0, 0, 3, 3, 3, 5, 5]],
        solution='Scanl1(lambda u1, u2: Max(u1, u2), x)',
    ),

    # Tasks using a combination of higher-order functions.
    ######################################################

    # TODO(kshi)
]


def run_program(program, inputs_dict):
  outputs = []
  num_examples = len(inputs_dict[next(iter(inputs_dict))])
  for i in range(num_examples):
    namespace = _OPS_NAMESPACE.copy()
    namespace.update({name: value[i] for name, value in inputs_dict.items()})
    outputs.append(eval(program, namespace))  # pylint: disable=eval-used
  return outputs


def simplify(program):
  """Replace `(lambda a: expr(a))(b)` with `expr(b)`."""
  program = re.sub(r'\s+', ' ', program.strip())
  program = re.sub(r'\s*\(\s*', '(', program)
  program = re.sub(r'\s*\)\s*', ')', program)
  program = re.sub(r':(?=[^ ])', ': ', program)
  print(program)

  changed = True
  while changed:  # Loop until there are no more changes.
    changed = False

    # One could do this with an AST walking approach, but we'd need third-party
    # tools to turn the AST back into code. Instead, let's just do it manually
    # with string parsing. Note that regex can't match balanced parentheses, so
    # we count those manually too.
    matching_paren_index = [None] * len(program)
    open_paren_index_stack = []
    for i, c in enumerate(program):
      if c == '(':
        open_paren_index_stack.append(i)
      elif c == ')':
        match_index = open_paren_index_stack.pop()
        matching_paren_index[match_index] = i
        matching_paren_index[i] = match_index
    assert not open_paren_index_stack

    # We want to find open parens followed by `lambda`, where the matching
    # closing paren is immediately followed by another pair of parens with a
    # variable list inside (and nothing else). We want to process the innermost
    # lambdas first, which will always appear after its outer lambdas, so we
    # start looking for this pattern from the end.
    for i, c in reversed(list(enumerate(program))):
      if c != '(':
        continue
      open_1 = i
      if not program[open_1 + 1:].startswith('lambda '):
        continue
      close_1 = matching_paren_index[i]
      open_2 = close_1 + 1
      if open_2 >= len(program) or program[open_2] != '(':
        continue
      close_2 = matching_paren_index[open_2]
      part_1 = program[open_1 + 1:close_1]
      match_1 = re.fullmatch(r'lambda ((?:\w+, )*\w+): (.*)', part_1)
      if not match_1:
        continue
      arg_names = match_1.group(1).split(', ')
      expression = match_1.group(2)
      print(f'arg_names: {arg_names}')
      print(f'expression: {expression}')
      part_2 = program[open_2 + 1:close_2]
      match_2 = re.fullmatch(r'(?:\w+, )*\w+', part_2)
      if not match_2:
        continue
      variable_names = match_2.group(0).split(', ')
      print(f'variable_names: {variable_names}')
      assert len(arg_names) == len(variable_names)
      renaming_dict = dict(zip(arg_names, variable_names))
      print(f'renaming_dict: {renaming_dict}')

      all_tokens = re.findall(r'\w+|\W+', expression)
      for token_i, token in enumerate(all_tokens):
        if token in renaming_dict:
          all_tokens[token_i] = renaming_dict[token]
      new_expression = ''.join(all_tokens)
      print(f'new_expression: {new_expression}')

      program = program[:open_1] + new_expression + program[close_2 + 1:]
      changed = True
      break  # Once we change one thing, the matching paren indices are stale.

  return program


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  inputs_dict = {
      'x': [1, 2, 3],
      'y': [10, 20, 30],
      'lst': [[10, 20], [-5], [3, 4, 5]]
  }
  # expected_output = [11, 22, 33]
  # program = 'Add(x, y)'
  expected_output = [[11, 21], [-3], [6, 7, 8]]
  program = 'Map(lambda u1: (lambda v1: Add(x, v1))(u1), lst)'
  actual_output = run_program(program, inputs_dict)
  print(actual_output)
  print(f'Matches? {actual_output == expected_output}')

  print(program)
  print(simplify(program))

  print()
  program = """
  ZipWith(
      lambda u1, u2:
          (lambda v1, v2:
              Add((lambda v1, v2: Subtract(v1, v2))(v1, v2),
                  1)
          )(u1, u2),
      lst1, lst2)
  """
  simple = simplify(program)
  print(program)
  print(simple)

  for task in HANDWRITTEN_TASKS:
    actual_output = run_program(task.solution, task.inputs_dict)
    success = actual_output == task.output
    print(f'Task {task.name} successful: {success}')
    if not success:
      print(f'Task: {task}')
      print(f'expected_output: {task.output}')
      print(f'actual_output:   {actual_output}')

if __name__ == '__main__':
  app.run(main)
