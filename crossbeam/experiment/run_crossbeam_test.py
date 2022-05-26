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

from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment import exp_common
from crossbeam.experiment import run_crossbeam
from crossbeam.experiment import train_eval

FLAGS = flags.FLAGS


class MainTupleTest(parameterized.TestCase):

  # This test is flaky.
  @parameterized.named_parameters(
      ('tuple', 'tuple', 'char', 5),
      ('arithmetic_char', 'arithmetic', 'char', 5),
      ('arithmetic_int', 'arithmetic', 'int', 5),
      ('bustle', 'bustle', 'char', 4))
  def test_crossbeam_memorizes(self, domain_str, model_type, max_weight):
    exp_common.set_global_seed(0)

    max_search_weight = max_weight + 1
    FLAGS([''])  # Parse flags
    FLAGS.domain = domain_str
    FLAGS.train_steps = 100
    FLAGS.eval_every = 50
    FLAGS.num_proc = 1
    FLAGS.lr = 0.01
    FLAGS.embed_dim = 32
    FLAGS.decoder_rnn_layers = 1
    FLAGS.max_search_weight = max_search_weight
    FLAGS.beam_size = 4
    FLAGS.grad_accumulate = 3    

    domain = domains.get_domain(domain_str)
    model = run_crossbeam.init_model(FLAGS, domain, model_type)

    proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
    eval_tasks = data_gen.gen_random_tasks(
        domain, num_tasks=4, min_weight=3, max_weight=max_weight,
        min_num_examples=2, max_num_examples=3,
        min_num_inputs=1, max_num_inputs=2)
    task_gen_func = lambda _: random.choice(eval_tasks)
    train_eval.main_train_eval(proc_args, model, eval_tasks,
                               task_gen=task_gen_func,
                               trace_gen=data_gen.trace_gen)

    success_rate, _ = train_eval.do_eval(
        eval_tasks, domain, model,
        max_search_weight=max_search_weight, beam_size=4, device='cpu',
        verbose=False)
    self.assertGreaterEqual(success_rate, 1/2)


if __name__ == '__main__':
  absltest.main()
