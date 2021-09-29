"""Tests for crossbeam.experiment.run_baseline_synthesizer."""

from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment import run_baseline_synthesizer


class BustleOperationsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tuple', 'tuple'),
      ('arithmetic', 'arithmetic'),
      ('bustle', 'bustle'))
  def test_run_synthesis(self, domain_str):
    domain = domains.get_domain(domain_str)
    num_tasks = 5
    tasks = data_gen.gen_random_tasks(
        domain, num_tasks=num_tasks, min_weight=3, max_weight=6,
        min_num_examples=2, max_num_examples=3,
        min_num_inputs=1, max_num_inputs=2, verbose=False)
    json_dict = run_baseline_synthesizer.run_synthesis(
        domain, tasks, timeout=1, verbose=False)
    self.assertEqual(sum(d['success'] for d in json_dict['results']), num_tasks)


if __name__ == '__main__':
  absltest.main()
