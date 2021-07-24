"""Converts old BUSTLE JSON files into pkl files."""
import json
import pickle as pkl

from absl import app
from absl import flags

from crossbeam.dsl import task as task_module

flags.DEFINE_string('input_json_file', None, 'JSON input filename')
flags.DEFINE_string('output_pkl_file', None, 'pkl output filename')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Loading JSON file {} ...'.format(FLAGS.input_json_file))
  with open(FLAGS.input_json_file, 'r') as f:
    json_data = json.load(f)

  assert isinstance(json_data, list)
  assert isinstance(json_data[0], dict)

  print('Processing {} tasks ...'.format(len(json_data)))
  tasks = []
  for json_task in json_data:
    examples = json_task['trainExamples']
    num_examples = len(examples)
    num_inputs = len(examples[0]['inputs'])
    inputs_dict = {'var_{}'.format(i):
                       [examples[e]['inputs'][i] for e in range(num_examples)]
                   for i in range(num_inputs)}
    outputs = [examples[e]['output'] for e in range(num_examples)]
    tasks.append(task_module.Task(inputs_dict, outputs, solution=None))

  print('Writing output file {} ...'.format(FLAGS.output_pkl_file))
  with open(FLAGS.output_pkl_file, 'wb') as f:
    pkl.dump(tasks, f, pkl.HIGHEST_PROTOCOL)

  print('Done!')

if __name__ == '__main__':
  app.run(main)
