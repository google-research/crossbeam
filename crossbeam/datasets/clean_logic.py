import os
import pickle5 as pkl

from crossbeam.datasets import logic_data

data_file = '~/data/crossbeam/logic_synthesis_10hr/train-tasks.pkl'

with open(os.path.expanduser(data_file), 'rb') as f:
  data = pkl.load(f)

print('Original data length: {}'.format(len(data)))

logic_constants, logic_operations = logic_data.get_consts_and_ops()
manual_tasks = logic_data.all_manual_logic_tasks(logic_operations)


solutions_to_indices = {task.solution: i for i, task in enumerate(data)}
bad_indices = []

for task in manual_tasks:
  manual_solution = task.solution
  if manual_solution in solutions_to_indices:
    index = solutions_to_indices[manual_solution]
    print('Found task in training data, index {}: {}'.format(index, task))
    bad_indices.append(index)

for i in sorted(bad_indices, reverse=True):
  print('Removing task index {}'.format(i))
  del data[i]

print('Filtered data length: {}'.format(len(data)))

with open('cleaned-train-tasks.pkl', 'wb') as f:
  pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
