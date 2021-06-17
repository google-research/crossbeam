"""Tests for crossbeam.datasets.bustle_data."""


from absl.testing import absltest
from absl.testing import parameterized

from crossbeam.datasets import bustle_data


class BustleDataTest(parameterized.TestCase):

  @parameterized.parameters(
      ('computer programs are cool', 'dynamic programming is neat', ' program'),
      ('123456', '1356234', '234'),
      ('aaaaa', 'a', 'a'),
      ('exact match', 'exact match', 'exact match'),
      ('aaaaa', 'b', ''),
      ('', '', ''))
  def test_compute_lcs(self, x, y, expected):
    self.assertEqual(bustle_data.compute_lcs(x, y), expected)

  def test_bustle_constants_extractor(self):
    inputs_dict = {'input_1': ["abcd", "wxyz", "0ab12"],
                   'input_2': ["$1234", "$5.67", "$89"]}
    outputs = ["01/01/01", "11/30/98", ""]

if __name__ == '__main__':
  absltest.main()
