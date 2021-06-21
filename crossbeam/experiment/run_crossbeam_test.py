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

  @parameterized.named_parameters(
      ('tuple', 'tuple', 'char', 5),
      ('arithmetic_char', 'arithmetic', 'char', 5),
      ('arithmetic_int', 'arithmetic', 'int', 5),
      ('bustle', 'bustle', 'char', 4))
  def test_crossbeam_memorizes(self, domain_str, model_type, max_weight):
    exp_common.set_global_seed(0)

    max_search_weight = max_weight + 1
    FLAGS([''])  # Parse flags
    FLAGS.train_steps = 150
    FLAGS.eval_every = 50
    FLAGS.num_proc = 1
    FLAGS.lr = 0.02
    FLAGS.embed_dim = 32
    FLAGS.decoder_rnn_layers = 1
    FLAGS.max_search_weight = max_search_weight
    FLAGS.beam_size = 4
    FLAGS.num_inputs = 1
    FLAGS.grad_accumulate = 2

    domain = domains.get_domain(domain_str)
    model = run_crossbeam.init_model(domain, model_type)

    proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
    eval_tasks = data_gen.gen_random_tasks(
        domain, 4, min_weight=3, max_weight=max_weight,
        num_examples=3, num_inputs=1)
    task_gen_func = lambda _: random.choice(eval_tasks)
    train_eval.main_train_eval(proc_args, model, eval_tasks, domain,
                               task_gen=task_gen_func,
                               trace_gen=data_gen.trace_gen)

    success_rate = train_eval.do_eval(
        eval_tasks, domain, model,
        max_search_weight=max_search_weight, beam_size=4, device='cpu',
        verbose=False)
    self.assertGreaterEqual(success_rate, 3/4)


if __name__ == '__main__':
  absltest.main()
