"""Creates a pkl file for handwritten DeepCoder tasks."""
import pickle5 as pkl

from crossbeam.dsl import task as task_module


def yield_tasks():
  """Yields handwritten DeepCoder tasks."""
  # map(lambda u: delta + u, lst)
  # Map(lambda u1: (lambda v1: Add(delta, v1))(u1), lst)
  # Weight 6.
  inputs_dict = {
      'delta': [3, 5],
      'lst': [[1, 2, 3], [4, 5, 6]],
  }
  outputs = [
      [4, 5, 6],
      [9, 10, 11],
  ]
  yield task_module.Task(inputs_dict, outputs, solution='')

  # filter(lambda u: u > limit, lst)
  # Filter(lambda u1: (lambda v1: Greater(v1, limit))(u1), lst)
  # Weight 6.
  inputs_dict = {
      'limit': [6, 10],
      'lst': [[4, 5, 6, 7, 8, 9], [20, 8, 10, 0, 11, 4, 9]],
  }
  outputs = [
      [7, 8, 9],
      [20, 11],
  ]
  yield task_module.Task(inputs_dict, outputs, solution='')

  # Scanl1(lambda u1, u2: (lambda v1, v2: Add(v1, v2))(u1, u2), xs)
  # Weight 7.
  inputs_dict = {
      'xs': [[3, 2, 6], [2, 5, 1]],
  }
  outputs = [
      [3, 5, 11],
      [2, 7, 8],
  ]
  yield task_module.Task(inputs_dict, outputs, solution='')

  # filter(lambda u: u > 2 * limit, lst)
  # Filter(lambda u1: (lambda v1: Greater(v1, Multiply(2, limit)))(u1), lst)
  # Weight 8.
  inputs_dict = {
      'limit': [3, 5],
      'lst': [[4, 5, 6, 7, 8, 9], [20, 8, 10, 0, 11, 4, 9]],
  }
  outputs = [
      [7, 8, 9],
      [20, 11],
  ]
  yield task_module.Task(inputs_dict, outputs, solution='')

  # ZipWith(lambda u1, u2: (lambda v1, v2: Multiply(v1, v2))(u1, u2), xs, ys)
  # Weight 8.
  inputs_dict = {
      'xs': [[3, 2, 6], [2, 5, 1]],
      'ys': [[2, 8, 3], [0, 2, 5]],
  }
  outputs = [
      [6, 16, 18],
      [0, 10, 5],
  ]
  yield task_module.Task(inputs_dict, outputs, solution='')


if __name__ == '__main__':
  with open('deepcoder_tasks.pkl', 'wb') as f:
    pkl.dump(list(yield_tasks()), f)
