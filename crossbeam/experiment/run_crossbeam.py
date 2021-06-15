# Copyright 2021 Google LLC
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

import argparse
import functools
import os
import pickle5 as cp
from absl import app
from absl import flags
import torch

from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import main_train_eval
from crossbeam.model.joint_model import JointModel, IntJointModel
from crossbeam.model.util import CharacterTable

FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'char', 'int/char')
flags.DEFINE_bool('stochastic_beam', False, 'do stochastic beam search during test')
flags.DEFINE_float('timeout', None, 'timeout during test')


def init_model(domain, model_type):
  """Initializes the model."""
  if model_type.startswith('char'):
    input_table = CharacterTable(domain.input_charset,
                                 max_len=domain.input_max_len)
    output_table = CharacterTable(domain.output_charset,
                                  max_len=domain.output_max_len)
    value_table = CharacterTable(domain.value_charset,
                                 max_len=domain.value_max_len)
    return JointModel(FLAGS, input_table, output_table, value_table,
                      domain.operations)
  elif model_type.startswith('int'):
    return IntJointModel(FLAGS,
                         input_range=(0, 10),
                         output_range=(-800, 800),
                         value_range=(-800, 800),
                         operations=domain.operations)
  else:
    raise ValueError('unknown model type %s' % model_type)


def main(argv):
  del argv
  set_global_seed(FLAGS.seed)

  domain = domains.get_domain(FLAGS.domain)
  model = init_model(domain, FLAGS.model_type)
  if FLAGS.load_model is not None:
    model_dump = os.path.join(FLAGS.save_dir, FLAGS.load_model)
    print('loading model from', model_dump)
    model.load_state_dict(torch.load(model_dump))
  if FLAGS.do_test:
    eval_file = 'test-tasks.pkl'
  else:
    eval_file = 'valid-tasks.pkl'
  with open(os.path.join(FLAGS.data_folder, eval_file), 'rb') as f:
    eval_tasks = cp.load(f)

  proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
  task_gen_func = functools.partial(
      data_gen.task_gen,
      min_weight=FLAGS.min_task_weight,
      max_weight=FLAGS.max_task_weight,
      num_examples=FLAGS.num_examples,
      num_inputs=FLAGS.num_inputs,
      verbose=FLAGS.verbose)
  main_train_eval(proc_args, model, eval_tasks, domain,
                  task_gen=task_gen_func,
                  trace_gen=data_gen.trace_gen)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)
