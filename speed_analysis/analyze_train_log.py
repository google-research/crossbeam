"""Analyze train_log.txt and plot data.

The log looks like:

{'elapsed_time': 0.2542449390166439,
 'explored_percent_concrete': 0.0,
 'explored_percent_lambda': 1.7543859649122806,
 'explored_percent_none': 98.24561403508773,
 'kept_percent_concrete': 0.0,
 'kept_percent_lambda': 100.0,
 'num_explored_concrete': 0,
 'num_explored_lambda': 10,
 'num_explored_none': 560,
 'num_kept_concrete': 0,
 'num_kept_lambda': 10,
 'num_unique_values': 23,
 'num_values_explored': 570,
 'num_values_kept': 10,
 'task_num_inputs': 2,
 'task_solution_weight': 7}
{'elapsed_time': 0.16423807095270604,
...
 'task_solution_weight': 7}
Grad step time: 1.16 sec
Synthesis time: 0.90 sec (77.3% of grad step time)
{'elapsed_time': 0.22671166399959475,
...
"""

import json
import os
import random

from absl import app
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def load_data(path):
  """Reads the train log from a path and returns a list of dicts."""
  data = []
  with open(os.path.expanduser(path)) as f:
    lines = f.readlines()
  i = 0
  while i < len(lines):
    line = lines[i]
    if line.startswith("{'elapsed_time':"):
      end_index = i + 1
      while not lines[end_index].endswith('}\n'):
        end_index += 1
      json_str = ''.join(lines[i : end_index + 1]).replace("'", '"')
      i = end_index + 1
      if '"stats":' in json_str:
        # This comes from evaluation, ignore it.
        continue
      data.append(json.loads(json_str))
    elif line.startswith('Grad '):
      grad_step_time = float(line.split(' ')[3])
      synthesis_time = float(lines[i + 1].split(' ')[2])
      i += 2
      data.append({'grad_step_time': grad_step_time,
                   'synthesis_time': synthesis_time,
                   'synthesis_percent': synthesis_time * 100 / grad_step_time})
    else:
      # Ignore other lines in the log such as eval info.
      i += 1
  return data


def shuffle_lists(*lists):
  zipped = list(zip(*lists))
  random.shuffle(zipped)
  return zip(*zipped)


def analyze_data(data, x_axis_1, y_axis_1, x_axis_2=None, y_axis_2=None):
  """Analyzes data to produce plots."""
  if x_axis_2 and y_axis_2:
    raise ValueError('Expected at most 1 of x_axis_2 and y_axis_2')

  # Collect datapoints from the raw data.
  xs_1, ys_1, indices_1 = [], [], []  # Data for main plot.
  xs_2, ys_2, indices_2 = [], [], []  # Data for secondary plot.
  for index, d in enumerate(data):
    # There are 4 synthesis searches per gradient update, so 5 data elements per
    # gradient update. Let's use `index` to track the number of gradient updates
    # so far, so synthesis search has a fractional "index" denoting partial
    # progress toward the next gradient update.
    index = (index + 1)/5

    # Main plot data.
    x_1 = index if x_axis_1 == 'index' else d.get(x_axis_1, None)
    y_1 = index if y_axis_1 == 'index' else d.get(y_axis_1, None)
    if x_1 is not None and y_1 is not None:
      xs_1.append(x_1)
      ys_1.append(y_1)
      indices_1.append(index)

    # Secondary plot data.
    if y_axis_2:
      x_2 = x_1
      y_2 = index if y_axis_2 == 'index' else d.get(y_axis_2, None)
    elif x_axis_2:
      x_2 = index if x_axis_2 == 'index' else d.get(x_axis_2, None)
      y_2 = y_1
    else:
      continue
    if x_2 is not None and y_2 is not None:
      xs_2.append(x_2)
      ys_2.append(y_2)
      indices_2.append(index)

  # Create a figure.
  plt.rcParams.update({'font.size': 15})
  fig, ax1 = plt.subplots()
  fig.set_size_inches(10, 7)

  # Shuffle the order the data is plotted in, to avoid later data always
  # covering earlier data.
  xs_1, ys_1, indices_1 = shuffle_lists(xs_1, ys_1, indices_1)
  if xs_2:
    xs_2, ys_2, indices_2 = shuffle_lists(xs_2, ys_2, indices_2)

  # Plot main scatterplot.
  cmap_1 = LinearSegmentedColormap.from_list(
      'cmap_1', ['blue', 'darkturquoise', 'limegreen', 'gold'])
  ax1.scatter(xs_1, ys_1, c=indices_1, marker='.', cmap=cmap_1, s=4)
  ax1.set_xlabel(x_axis_1)
  ax1.set_ylabel(y_axis_1)
  ax1.set_axisbelow(True)
  ax1.grid(color='lightgray')

  # Plot a fit curve for main scatterplot.
  polynomial = np.poly1d(np.polyfit(xs_1, ys_1, 5))
  lower, upper = np.percentile(xs_1, [1, 99])
  fit_linspace = np.linspace(lower, upper, num=100)
  ax1.plot(fit_linspace, polynomial(fit_linspace), color='red')

  # Plot a secondary scatterplot.
  if y_axis_2:
    ax2 = ax1.twinx()
    ax2.set_ylabel(y_axis_2)
  elif x_axis_2:
    ax2 = ax1.twiny()
    ax2.set_xlabel(x_axis_2)
  else:
    ax2 = None
  if ax2:
    cmap_2 = LinearSegmentedColormap.from_list(
        'cmap_2', ['salmon', 'darkorange', 'saddlebrown'])
    ax2.scatter(xs_2, ys_2, c=indices_2, marker='.', cmap=cmap_2, s=2)
    # Make sure the secondary plot doesn't cover the main one.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)

  # Save the figure.
  fig.savefig('plot.png')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load data.
  data = load_data('~/xlambda/train_log_2.txt')

  # Reference for axis choices:
  _ = """
{'elapsed_time': 0.12969667906872928,
 'explored_percent_concrete': 47.36842105263158,
 'explored_percent_lambda': 0.0,
 'explored_percent_none': 52.63157894736842,
 'kept_percent_concrete': 100.0,
 'kept_percent_lambda': 0.0,
 'num_examples': 4,
 'num_explored_concrete': 270,
 'num_explored_lambda': 0,
 'num_explored_none': 300,
 'num_inputs': 2,
 'num_kept_concrete': 23,
 'num_kept_lambda': 0,
 'num_unique_values': 36,
 'num_values_explored': 570,
 'num_values_kept': 23,
 'task_num_inputs': 2,
 'task_solution_weight': 7}
and
'grad_step_time'
'synthesis_time'
'synthesis_percent'
'index'
"""

  # Axis labels for main plot.
  x_axis_1 = 'num_values_kept'
  y_axis_1 = 'elapsed_time'

  # Axis labels for a secondary plot. Only use one of x_axis_2 or y_axis_2.
  x_axis_2 = ''
  y_axis_2 = ''
  assert not x_axis_2 or not y_axis_2

  # Create plot.
  analyze_data(data, x_axis_1, y_axis_1, x_axis_2, y_axis_2)


if __name__ == '__main__':
  app.run(main)
