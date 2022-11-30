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

"""Tests for crossbeam.experiment.run_crossbeam."""

import argparse
import random

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.algorithm import variables as variables_module
from crossbeam.datasets import data_gen
from crossbeam.dsl import deepcoder_operations
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module
from crossbeam.experiment import exp_common
from crossbeam.experiment import run_crossbeam
from crossbeam.experiment import train_eval

FLAGS = flags.FLAGS


def task_1():
  inputs_dict = {
      'x': [1, 2, 3],
      'y': [10, 20, 30],
  }
  outputs = [11, 22, 33]
  input_vars = {name: value_module.InputVariable(values, name)
                for name, values in inputs_dict.items()}
  task = task_module.Task(
      inputs_dict=inputs_dict,
      outputs=outputs,
      solution=deepcoder_operations.Add().apply(
          [input_vars['x'], input_vars['y']]))
  return task, 'Add(x, y)'


def task_2():
  inputs_dict = {
      'x': [1, 2, 3],
      'y': [10, 20, 30],
  }
  outputs = [10, 40, 90]
  input_vars = {name: value_module.InputVariable(values, name)
                for name, values in inputs_dict.items()}
  task = task_module.Task(
      inputs_dict=inputs_dict,
      outputs=outputs,
      solution=deepcoder_operations.Multiply().apply(
          [input_vars['x'], input_vars['y']]))
  return task, 'Multiply(x, y)'


def task_3():
  inputs_dict = {
      'xs': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
      'delta': [30, 20, 10],
  }
  outputs = [[31, 32, 33], [24, 25], [16, 17, 18, 19]]
  input_vars = {name: value_module.InputVariable(values, name)
                for name, values in inputs_dict.items()}
  v1 = variables_module.first_free_vars(1)[0]
  u1 = variables_module.first_bound_vars(1)[0]
  task = task_module.Task(
      inputs_dict=inputs_dict,
      outputs=outputs,
      solution=deepcoder_operations.Map().apply(
          [deepcoder_operations.Add().apply([v1, input_vars['delta']],
                                            free_variables=[v1]),
           input_vars['xs']],
          arg_variables=[[u1], []]))
  return task, 'Map(lambda u1: (lambda v1: Add(v1, delta))(u1), xs)'


def task_4():
  inputs_dict = {
      'xs': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
      'delta': [30, 20, 10],
  }
  outputs = [[30, 60, 90], [80, 100], [60, 70, 80, 90]]
  input_vars = {name: value_module.InputVariable(values, name)
                for name, values in inputs_dict.items()}
  v1 = variables_module.first_free_vars(1)[0]
  u1 = variables_module.first_bound_vars(1)[0]
  task = task_module.Task(
      inputs_dict=inputs_dict,
      outputs=outputs,
      solution=deepcoder_operations.Map().apply(
          [deepcoder_operations.Multiply().apply([v1, input_vars['delta']],
                                                 free_variables=[v1]),
           input_vars['xs']],
          arg_variables=[[u1], []]))
  return task, 'Map(lambda u1: (lambda v1: Multiply(v1, delta))(u1), xs)'


class RunCrossBeamTest(parameterized.TestCase):

  # This test is flaky and long-running.
  def test_crossbeam_memorizes(self):
    exp_common.set_global_seed(0)

    max_search_weight = 8
    FLAGS([''])  # Parse flags
    FLAGS.save_dir = '/tmp/crossbeam/'
    FLAGS.domain = 'deepcoder'
    FLAGS.train_steps = 100
    FLAGS.eval_every = 20  # Run pytest with the `-s` flag to see training logs.
    FLAGS.num_proc = 1
    FLAGS.lr = 0.005
    FLAGS.embed_dim = 32
    FLAGS.decoder_rnn_layers = 1
    FLAGS.max_search_weight = max_search_weight
    FLAGS.beam_size = 4
    FLAGS.grad_accumulate = 3
    FLAGS.io_encoder = 'lambda_signature'
    FLAGS.value_encoder = 'lambda_signature'
    FLAGS.use_ur = False

    domain = domains.get_domain('deepcoder')
    model = run_crossbeam.init_model(FLAGS, domain, 'deepcoder')

    proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
    tasks_with_solutions = [task_1(), task_2(), task_3(), task_4()]

    for task, solution_expr in tasks_with_solutions:
      self.assertEqual(task.solution.expression(), solution_expr)
      self.assertEqual(task.solution, value_module.OutputValue(task.outputs))

    eval_tasks = [task for task, _ in tasks_with_solutions]

    task_gen_func = lambda _: random.choice(eval_tasks)
    train_eval.main_train_eval(proc_args, model, eval_tasks,
                               task_gen=task_gen_func,
                               trace_gen=data_gen.trace_gen)

    success_rate, _ = train_eval.do_eval(
        eval_tasks, domain, model,
        max_search_weight=max_search_weight, beam_size=4, device='cpu',
        use_ur=False, verbose=False)
    self.assertEqual(success_rate, 1)


if __name__ == '__main__':
  absltest.main()
