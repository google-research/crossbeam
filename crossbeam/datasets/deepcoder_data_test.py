"""Tests for crossbeam.datasets.deepcoder_data_test."""

import itertools
import random

from absl.testing import absltest

from crossbeam.datasets import deepcoder_data


class BottomUpDataGenerationTest(absltest.TestCase):

  def test_deepcoder_inputs_dict_generator(self):
    for _ in range(100):
      num_inputs = random.randint(1, 3)
      num_examples = random.randint(2, 5)
      inputs_dict = deepcoder_data.deepcoder_inputs_dict_generator(
          num_inputs=num_inputs, num_examples=num_examples)

      # No two input variables are identical.
      for name_1, name_2 in itertools.combinations(inputs_dict.keys(), 2):
        self.assertNotEqual(inputs_dict[name_1], inputs_dict[name_2])

      # No two examples are identical.
      for index_1, index_2 in itertools.combinations(range(num_examples), 2):
        example_1 = {name: values[index_1]
                     for name, values in inputs_dict.items()}
        example_2 = {name: values[index_2]
                     for name, values in inputs_dict.items()}
        self.assertNotEqual(example_1, example_2)

if __name__ == '__main__':
  absltest.main()
