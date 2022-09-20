"""Evaluate property signature quality with heuristics.

Original:

Performing evaluation on property signatures.
value_set has 3458 elements:
  * 2256 concrete values
  * 1202 lambda values

Computed 2256 signatures for concrete values in 0.569 total seconds, or 0.00025 seconds each.

Computed 1202 signatures for lambda values in 2.201 total seconds, or 0.00183 seconds each.

Signature length: 1114

Of the lambda values, 678 fail to execute.
We are ignoring these and keeping the other 524 values that do execute.
  Removing key [[]] (394 values)...
    Example value: lambda v1: Last((lambda v1: Maximum(v1))(v1))
  Removing key [[], []] (284 values)...
    Example value: lambda v1: If(v1, -1, lst)
The following lambdas have the same functionality: [[((-100,), 3), ((-50,), 3), ((-20,), 3), ((-10,), 3), ((-7,), 3), ((-5,), 3), ((-4,), 3), ((-3,), 3
  lambda v1: Max(delta, v1)
  lambda v1: Max(v1, delta)
The following lambdas have the same functionality: [[(([0],), 0), (([-1],), -1), (([1],), 1), (([6],), 6), (([0, 1],), 1), (([2, 2],), 2), (([-2, -4],)
  lambda v1: Head((lambda v1: Reverse(v1))(v1))
  lambda v1: Last(v1)
Found 203 lambdas with duplicate functionality.
We are ignoring those and keeping the other 321 values with unique functionality.

For concrete signatures:
Values have the same signature:
  [int:-8, int:-8]
  [int:-7, int:-9]
Values have the same signature:
  [int:-8, int:-8]
  [int:-9, int:-9]
Min distance: 0.0
Mean distance: 7.234567245538306
Max distance: 13.96424004376894
Number of zero-distance pairs: 27008 (1.06%)

For lambda signatures:
Values have the same signature:
  lambda v1: Less(Square(delta), v1)
  lambda v1: Less(Square(4), v1)
Values have the same signature:
  lambda v1: Less(v1, delta)
  lambda v1: Greater(4, v1)
Min distance: 0.0
Mean distance: 11.33844540785091
Max distance: 22.94396908869914
Number of zero-distance pairs: 69 (0.13%)

"""

import collections
import itertools
import timeit
from absl import app

from crossbeam.algorithm import baseline_enumeration
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module
from crossbeam.property_signatures import property_signatures
# from crossbeam.property_signatures import old_property_signatures as property_signatures

import numpy as np

VERBOSITY = 2


def sig_to_np(sig):
  return np.array([[frac_applicable, frac_true]
                   for frac_applicable, _, _, frac_true in sig])


def sig_distance(x, y):
  return np.linalg.norm(x - y)


def analyze_distances(values, sigs):
  """Analyzes distances between signatures."""
  assert len(values) == len(sigs)

  distances = []
  num_printed = 0
  for i, j in itertools.combinations(range(len(values)), r=2):
    distance = sig_distance(sigs[i], sigs[j])
    distances.append(distance)
    if distance == 0 and num_printed < VERBOSITY:
      print('Values have the same signature:')
      print(f'  {values[i]}')
      print(f'  {values[j]}')
      num_printed += 1

  print(f'Min distance: {min(distances)}')
  print(f'Mean distance: {np.mean(distances)}')
  print(f'Max distance: {max(distances)}')
  num_zero_distance = sum(d == 0 for d in distances)
  print(f'Number of zero-distance pairs: {num_zero_distance} '
        f'({num_zero_distance / len(distances) * 100:.2f}%)')


def evaluate_property_signatures(value_set, output_value):
  """Evaluates how different property signatures are for the set of values."""
  print('\n' + '=' * 80)
  print('\nPerforming evaluation on property signatures.')
  concrete_values = [x for x in value_set if not x.num_free_variables]
  lambda_values = [x for x in value_set if x.num_free_variables]
  print(f'value_set has {len(value_set)} elements:\n'
        f'  * {len(concrete_values)} concrete values\n'
        f'  * {len(lambda_values)} lambda values')
  print()

  start_time = timeit.default_timer()
  concrete_sigs = [
      property_signatures.property_signature_value(v, output_value)
      for v in concrete_values]
  elapsed_time = timeit.default_timer() - start_time
  print(f'Computed {len(concrete_sigs)} signatures for concrete values in '
        f'{elapsed_time:.3f} total seconds, or '
        f'{elapsed_time/len(concrete_sigs):.5f} seconds each.')
  print()

  start_time = timeit.default_timer()
  lambda_sigs = [
      property_signatures.property_signature_value(v, output_value)
      for v in lambda_values]
  elapsed_time = timeit.default_timer() - start_time
  print(f'Computed {len(lambda_sigs)} signatures for lambda values in '
        f'{elapsed_time:.3f} total seconds, or '
        f'{elapsed_time/len(lambda_sigs):.5f} seconds each.')
  print()

  print(f'Signature length: {len(lambda_sigs[0])}')
  assert len(set(len(s) for s in concrete_sigs + lambda_sigs)) == 1
  print()

  lambda_functionality_dict = collections.defaultdict(list)
  failed_execution_key_set = set()
  for v, sig in zip(lambda_values, lambda_sigs):
    io_pairs_per_example = property_signatures._run_lambda(v)  # pylint: disable=protected-access
    key = str(io_pairs_per_example)
    lambda_functionality_dict[key].append((v, sig))
    if all(not pairs_for_example for pairs_for_example in io_pairs_per_example):
      # The lambda doesn't execute successfully for any attempted input list.
      failed_execution_key_set.add(key)

  if failed_execution_key_set:
    num_failed = sum(len(lambda_functionality_dict[key])
                     for key in failed_execution_key_set)
    print(f'Of the lambda values, {num_failed} fail to execute.')
    print(f'We are ignoring these and keeping the other '
          f'{len(lambda_values) - num_failed} values that do execute.')
    for key in failed_execution_key_set:
      print(f'  Removing key {key} ({len(lambda_functionality_dict[key])} '
            f'values)...')
      print(f'    Example value: {lambda_functionality_dict[key][0][0]}')
      del lambda_functionality_dict[key]

  lambda_values = []
  lambda_sigs = []
  num_duplicate_functionality = 0
  num_printed = 0
  for key, values in lambda_functionality_dict.items():
    first_value, first_sig = values[0]
    lambda_values.append(first_value)
    lambda_sigs.append(first_sig)
    num_duplicate_functionality += len(values) - 1
    if len(values) > 1 and num_printed < VERBOSITY:
      print(f'The following lambdas have the same functionality: {key[:100]}')
      for v, _ in values[:VERBOSITY]:
        print(f'  {v}')
      num_printed += 1
  print(f'Found {num_duplicate_functionality} lambdas with duplicate '
        f'functionality.')
  print(f'We are ignoring those and keeping the other {len(lambda_values)} '
        f'values with unique functionality.')
  print()

  concrete_sigs_np = [sig_to_np(sig) for sig in concrete_sigs]
  lambda_sigs_np = [sig_to_np(sig) for sig in lambda_sigs]

  print('For concrete signatures:')
  analyze_distances(concrete_values, concrete_sigs_np)
  print()

  print('For lambda signatures:')
  analyze_distances(lambda_values, lambda_sigs_np)
  print()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  inputs_dict = {
      'delta': [3, 5],
      'lst': [[1, 2, 3], [4, 5, 6]],
  }
  outputs = [
      [4, 5, 6],
      [9, 10, 11],
  ]
  task = task_module.Task(inputs_dict, outputs, solution='')
  domain = domains.get_domain('deepcoder')
  result, value_set, _, _ = baseline_enumeration.synthesize_baseline(
      task, domain, timeout=5)
  assert (result.expression() ==
          'Map(lambda u1: (lambda v1: Add(delta, v1))(u1), lst)')

  evaluate_property_signatures(value_set, value_module.OutputValue(outputs))


if __name__ == '__main__':
  app.run(main)
