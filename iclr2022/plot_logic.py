"""Plots the success rate by the size of logic tasks.

python3 plot_logic.py --output_file=logic_success_rate_plot.png

"""

import collections
import glob
import json
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pkl
matplotlib.use('Agg')
import seaborn as sns  # pylint: disable=g-import-not-at-top

FLAGS = flags.FLAGS

flags.DEFINE_string('output_file', 'logic_success_rate_plot.png',
                    'file where we will save the output plot.')

flags.DEFINE_string('baseline_pkl_file', 'logic_baseline_data.pkl',
                    'Pickle file containing results for logic baselines.')

MIN_TASK_SIZE = 5
MAX_TASK_SIZE = 19
TIME_LIMIT = 30  # Time limit in seconds.

NUM_TASKS_PER_SIZE = collections.defaultdict(lambda: 50)
NUM_TASKS_PER_SIZE[5] = 13
NUM_TASKS_PER_SIZE[6] = 40

# Directory to find CrossBeam results.
CROSSBEAM_DIR = 'logic_results'

# Key -> (label, color, linestyle, linewidth, marker, markersize).
LINE_FORMATS = {
    'm0': ('Metagol (Our DSL)', 'orchid', '--', 1.5, '+', 10),
    'm1': ('Metagol (README DSL)', 'blueviolet', ':', 1.5, 'x', 7),
    'p0': ('Popper (Pred. Inv.)', 'darkorange', '-', 1.5, 'v', 7),
    'p1': ('Popper (No Pred. Inv.)', 'darkred', '-.', 1.5, '^', 7),
}

# CrossBeam file formats ->
# (label, color, linestyle, linewidth, border_width, marker, markersize)
CROSSBEAM_FORMATS = {
    'run_*.sat-new-transformer.logic_by_weight.json':
        ('CrossBeam (Transformer)', '#028a0f', '-', 2, 1, '*', 10),  # green
    'run_*.sat-new-mlp.logic_by_weight.json':
        ('CrossBeam (MLP)', 'dodgerblue', '--', 1.5, 1, 'D', 5),
}

# File format and line format for baseline.
BASELINE_FILE_FORMAT = 'baseline.logic_by_weight.size-{}.json'
BASELINE_SIZES = list(range(5, 15))
BASELINE_LINE_FORMAT = ('Baseline Bottom-Up Search', 'black', '-', 1.5, 'o', 6)

# Plot title.
TITLE = 'Success Rate on Random Logic Tasks of Varying Size'


def create_plot(baseline_pkl_file):
  """Plots benchmarks solved per expression considered."""
  xs = np.arange(MIN_TASK_SIZE, MAX_TASK_SIZE + 1)
  legend_handles = []

  # CrossBeam lines.
  for file_format, (label, color, linestyle, linewidth, border_width,
                    marker, markersize) in CROSSBEAM_FORMATS.items():
    all_runs = []
    solve_times_by_size = collections.defaultdict(list)
    for filename in sorted(glob.glob(os.path.join(CROSSBEAM_DIR, file_format))):
      print('Reading CrossBeam results file: {}'.format(filename))
      with open(filename) as f:
        this_data = json.load(f)

      solves_by_size = [0] * (MAX_TASK_SIZE + 1)
      for result in this_data['results']:
        if result['success'] and result['elapsed_time'] < TIME_LIMIT:
          size = result['task_solution_weight']
          solve_times_by_size[size].append(result['elapsed_time'])
          solves_by_size[size] += 1
      success_rate_by_size = [
          100 * solves_by_size[size] / NUM_TASKS_PER_SIZE[size] for size in xs]
      all_runs.append(success_rate_by_size)

    all_runs = np.array(all_runs)
    all_runs_max = np.max(all_runs, axis=0)
    all_runs_min = np.min(all_runs, axis=0)
    all_runs_mean = np.mean(all_runs, axis=0)

    line, = plt.plot(xs, all_runs_mean, label=label, color=color,
                     linestyle=linestyle, linewidth=linewidth,
                     marker=marker, markersize=markersize)

    fill = plt.fill_between(xs, all_runs_min, all_runs_max, facecolor=color,
                            alpha=0.1)
    plt.plot(xs, all_runs_max, label=None, color=color, alpha=0.3,
             linestyle='-', linewidth=border_width)
    plt.plot(xs, all_runs_min, label=None, color=color, alpha=0.3,
             linestyle='-', linewidth=border_width)
    legend_handles.append((line, fill))

    print('For method {}:'.format(label))
    for size, rate in zip(xs, all_runs_mean):
      print('  Size {}: solved {:.1f}% of tasks in {:.2f} sec on average'
            .format(size, rate, np.mean(solve_times_by_size[size])))

  # Line for baseline bottom-up enumerative search.
  solves_by_size = {}
  solve_times_by_size = {}
  for size in BASELINE_SIZES:
    filename = os.path.join(CROSSBEAM_DIR, BASELINE_FILE_FORMAT.format(size))
    print('Reading baseline results file: {}'.format(filename))
    with open(filename) as f:
      data_for_size = json.load(f)
    solves_by_size[size] = sum(
        result['success'] and result['elapsed_time'] < TIME_LIMIT
        for result in data_for_size['results'])
    solve_times_by_size[size] = np.mean(
        [r['elapsed_time'] for r in data_for_size['results']
         if r['success'] and r['elapsed_time'] < TIME_LIMIT])

  success_rate_by_size = []
  for size in xs:
    if size < min(BASELINE_SIZES):
      success_rate_by_size.append(100)
    elif size > max(BASELINE_SIZES):
      success_rate_by_size.append(0)
    else:
      success_rate_by_size.append(
          100 * solves_by_size[size] / NUM_TASKS_PER_SIZE[size])
  label, color, linestyle, linewidth, marker, markersize = BASELINE_LINE_FORMAT
  line, = plt.plot(xs, success_rate_by_size, label=label, color=color,
                   linestyle=linestyle, linewidth=linewidth,
                   marker=marker, markersize=markersize)
  legend_handles.append(line)

  print('Baseline solves_by_size: {}'.format(solves_by_size))
  print('For Baseline:')
  for size, rate in zip(xs, success_rate_by_size):
    print('  Size {}: solved {:.1f}% of tasks in {:.2f} sec on average'.format(
        size, rate, solve_times_by_size.get(size, float('nan'))))
    if rate == 0:
      break

  # Lines for logic baselines.
  with open(os.path.expanduser(baseline_pkl_file), 'rb') as f:
    baseline_results = pkl.load(f)

  for key, (label, color, linestyle, linewidth,
            marker, markersize) in LINE_FORMATS.items():
    results = baseline_results[key]
    solves_by_size = [sum(bool(0 < time < TIME_LIMIT) for time in results[i])
                      for i in range(len(results))]
    solve_times_by_size = [np.mean([time for time in results[i]
                                    if 0 < time < TIME_LIMIT])
                           for i in range(len(results))]
    success_rate_by_size = [
        100 * solves_by_size[size] / NUM_TASKS_PER_SIZE[size] for size in xs]

    line, = plt.plot(xs, success_rate_by_size, label=label, color=color,
                     linestyle=linestyle, linewidth=linewidth, marker=marker,
                     markersize=markersize)
    legend_handles.append(line)

    print('For method {}:'.format(label))
    for size, rate in zip(xs, success_rate_by_size):
      print('  Size {}: solved {:.1f}% of tasks in {:.2f} sec on average'
            .format(size, rate, solve_times_by_size[size]))
      if rate == 0:
        break

  plt.xlim(MIN_TASK_SIZE - 0.25, MAX_TASK_SIZE + 0.25)
  plt.xticks(xs)
  plt.ylim(-2, 102)
  plt.title(TITLE, fontsize=14)
  plt.xlabel('Task Size', fontsize=12)
  plt.ylabel('Success Rate (%)', fontsize=12)

  _, labels = plt.gca().get_legend_handles_labels()
  handles = legend_handles
  legend = plt.legend(handles, labels,
                      loc='lower left', bbox_to_anchor=(1.04, 0))
  return legend


def main(_):
  sns.set()

  default_width = 6.4
  default_height = 4.8
  plt.figure(figsize=(default_width, default_height * 2/3))

  legend = create_plot(baseline_pkl_file=FLAGS.baseline_pkl_file)

  plt.savefig(
      os.path.expanduser(FLAGS.output_file),
      bbox_inches='tight',
      bbox_extra_artists=[legend])
  plt.clf()


if __name__ == '__main__':
  app.run(main)
