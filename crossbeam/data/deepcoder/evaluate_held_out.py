"""Evaluate solutions on held-out test cases."""

from collections.abc import Sequence
import contextlib
import functools
import json
import os
import signal

from absl import app

from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.dsl import deepcoder_utils


RESULTS_JSON = 'comparisons/results/synthetic/baseline_enumeration_timeout600.json'
HELD_OUT_JSON = 'crossbeam/data/deepcoder/held_out_test_cases.json'

IS_LLM = 'PaLM' in RESULTS_JSON


class Timeout:
  """Context manager for setting a timeout."""

  def __init__(self, seconds=1, error_message='Timeout'):
    self.seconds = seconds
    self.error_message = error_message

  def handle_timeout(self, signum, frame):
    raise TimeoutError(self.error_message)

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, self.seconds)

  def __exit__(self, type_, value, traceback):
    signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def suppress_stdout():
  with open(os.devnull, 'w') as null:
    with contextlib.redirect_stdout(null):
      yield


def name_from_task_str(task_str):
  return task_str.split("'")[1]


@functools.cache
def get_held_out_tasks():
  with open(HELD_OUT_JSON) as f:
    return json.load(f)


def evaluate_result(result, is_llm, verbose=False):
  """Evaluates a result (solution) using held-out test cases."""
  solution = result['solution']
  if is_llm:
    task_name = result['name']
  else:
    solution = deepcoder_utils.simplify(solution)
    task_name = name_from_task_str(result['task'])
  if task_name in deepcoder_tasks.RENAMING_MAP:
    new_name = deepcoder_tasks.RENAMING_MAP[task_name]
    solution = solution.replace(task_name, new_name)
    task_name = new_name

  held_out_task = [task for task in get_held_out_tasks()
                   if task['name'] == task_name]
  assert len(held_out_task) == 1
  held_out_task = held_out_task[0]
  held_out_inputs_dict = held_out_task['held_out_inputs_dict']
  held_out_outputs = held_out_task['held_out_outputs']

  success = True
  try:
    if is_llm:
      assert ':' in task_name
      function_name = task_name.split(':')[1]
      using_nl = f'def {function_name}(' in solution
      if not using_nl:
        assert 'def func(x1' in solution

      namespace = {}
      with Timeout(seconds=0.1), suppress_stdout():
        exec(solution, namespace)  # pylint: disable=exec-used
        num_inputs = len(held_out_inputs_dict)
        num_examples = len(held_out_outputs)
        input_names = (list(held_out_inputs_dict) if using_nl else
                       [f'x{i + 1}' for i in range(num_inputs)])
        code_to_eval = (f'{function_name if using_nl else "func"}('
                        f'{",".join(input_names)})')
        for example_index in range(num_examples):
          for input_name, input_value in zip(input_names,
                                             held_out_inputs_dict.values()):
            namespace[input_name] = input_value[example_index]
          output = eval(code_to_eval, namespace)  # pylint: disable=eval-used
          if output != held_out_outputs[example_index]:
            success = False
            break
    else:
      outputs = deepcoder_utils.run_program(result['solution'],
                                            held_out_inputs_dict)
      success = outputs == held_out_outputs
  except Exception:  # pylint: disable=broad-except
    success = False

  if verbose and not success:
    if is_llm:
      print(f'False positive for {task_name}:\n{solution}\n')
    else:
      print(f'False positive for {task_name}: {solution}')
  return success


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(RESULTS_JSON) as f:
    results = json.load(f)['results']

  num_fail = 0
  num_success = 0
  num_false_positive = 0

  for result in results:

    if not result['success'] or result['elapsed_time'] > 600:
      num_fail += 1
      continue

    success = evaluate_result(result, is_llm=IS_LLM, verbose=True)

    if success:
      num_success += 1
    else:
      num_false_positive += 1

  print(f'num_fail: {num_fail}')
  print(f'num_success: {num_success}')
  print(f'num_false_positive: {num_false_positive}')


if __name__ == '__main__':
  app.run(main)
