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
from absl import logging
import torch
import random
from ml_collections import config_flags
from crossbeam.datasets import data_gen
from crossbeam.common import configs_all
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import main_train_eval
from crossbeam.model.joint_model import JointModel, IntJointModel
from crossbeam.model.logic_model import LogicModel
from crossbeam.model.deepcoder_model import DeepCoderModel
from crossbeam.model.util import CharacterTable

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string='Path to the Training configuration.')

flags.DEFINE_string('model_type', 'char', 'int/char/logic')
flags.DEFINE_bool('stochastic_beam', False, 'do stochastic beam search during test')
flags.DEFINE_bool('random_beam', False, 'replace beam search with random choices?')


def init_model(args, domain, model_type):
  """Initializes the model."""
  if model_type.startswith('char'):
    input_table = CharacterTable(domain.input_charset,
                                 max_len=domain.input_max_len)
    output_table = CharacterTable(domain.output_charset,
                                  max_len=domain.output_max_len)
    value_table = CharacterTable(domain.value_charset,
                                 max_len=domain.value_max_len)
    return JointModel(args, input_table, output_table, value_table,
                      domain.operations)
  elif model_type.startswith('int'):
    return IntJointModel(args,
                         input_range=(0, 10),
                         output_range=(-800, 800),
                         value_range=(-800, 800),
                         operations=domain.operations)
  elif model_type.startswith('logic'):
    return LogicModel(args, operations=domain.operations)
  elif model_type == 'deepcoder':
    return DeepCoderModel(args, operations=domain.operations)
  else:
    raise ValueError('unknown model type %s' % model_type)


def main(argv):
  del argv
  if FLAGS.config is None:
    config = FLAGS
    proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
  else:
    config = configs_all.get_config()
    config.update(FLAGS.config)
    proc_args = config
    config.data_folder = os.path.join(config.data_root, config.data_name)
  logging.info(proc_args)
  set_global_seed(config.seed)

  domain = domains.get_domain(config.domain)
  model = init_model(config, domain, config.model_type)
  if config.load_model is not None:
    model_dump = os.path.join(config.save_dir, config.load_model)
    print('loading model from', model_dump)
    model.load_state_dict(torch.load(model_dump))
    print('model loaded.')
  if config.do_test:
    eval_prefix = 'test-tasks'
  else:
    eval_prefix = 'valid-tasks'
  eval_files = os.listdir(config.data_folder)
  eval_files = [fname for fname in eval_files if fname.startswith(eval_prefix)]
  eval_tasks = []
  for fname in sorted(eval_files):
    with open(os.path.join(config.data_folder, fname), 'rb') as f:
      eval_tasks += cp.load(f)
    # Shuffle the evaluation tasks so that when we take the first `num_valid`
    # tasks, they come from different data-generation searches.
    random.shuffle(eval_tasks)

  if config.train_data_glob is not None:
    task_gen_func = None
  else:
    task_gen_func = functools.partial(
        data_gen.task_gen,
        min_weight=config.min_task_weight,
        max_weight=config.max_task_weight,
        min_num_examples=config.min_num_examples,
        max_num_examples=config.max_num_examples,
        min_num_inputs=config.min_num_inputs,
        max_num_inputs=config.max_num_inputs,
        verbose=config.verbose)
  
  main_train_eval(proc_args, model, eval_tasks,
                  task_gen=task_gen_func,
                  trace_gen=data_gen.trace_gen)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)
