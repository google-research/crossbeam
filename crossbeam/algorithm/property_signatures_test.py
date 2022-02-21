# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for property_signatures."""

from absl.testing import absltest

from crossbeam.algorithm import property_signatures
from crossbeam.algorithm.property_signatures import PropertySummary
from crossbeam.dsl import value as value_module


class PropertySignaturesTest(absltest.TestCase):

  def test_compute_example_signature(self):
    inputs = [value_module.InputValue(['butter', 'abc', 'xyz'], 'input_1')]
    output = value_module.OutputValue(['butterfly', 'abc_', 'XYZ_'])

    signature = property_signatures.compute_example_signature(inputs, output)

    # String properties of the input
    self.assertEqual(signature[0 : 5], [
        PropertySummary.ALL_FALSE,  # is empty?
        PropertySummary.ALL_FALSE,  # is single char?
        PropertySummary.MIXED,  # is short string?
        PropertySummary.ALL_TRUE,  # is lowercase?
        PropertySummary.ALL_FALSE,  # is uppercase?
    ])

    # Int properties of the input
    start_index = property_signatures.NUM_STRING_PROPERTIES
    end_index = start_index + property_signatures.NUM_INT_PROPERTIES
    self.assertEqual(
        signature[start_index : end_index],
        ([PropertySummary.TYPE_MISMATCH] *
         property_signatures.NUM_INT_PROPERTIES))

    # String properties of the input compared to the output
    start_index = property_signatures.NUM_SINGLE_VALUE_PROPERTIES
    self.assertEqual(signature[start_index : start_index + 9], [
        PropertySummary.MIXED,  # output contains input?
        PropertySummary.MIXED,  # output starts with input?
        PropertySummary.ALL_FALSE,  # output ends with input?
        PropertySummary.ALL_FALSE,  # input contains output?
        PropertySummary.ALL_FALSE,  # input starts with output?
        PropertySummary.ALL_FALSE,  # input ends with output?
        PropertySummary.ALL_TRUE,  # output contains input ignoring case?
        PropertySummary.ALL_TRUE,  # output starts with input ignoring case?
        PropertySummary.ALL_FALSE,  # output ends with input ignoring case?
    ])

    # Int properties of the input compared to the output
    start_index = (property_signatures.NUM_SINGLE_VALUE_PROPERTIES +
                   property_signatures.NUM_STRING_COMPARISON_PROPERTIES)
    end_index = start_index + property_signatures.NUM_INT_COMPARISON_PROPERTIES
    self.assertEqual(
        signature[start_index : end_index],
        ([PropertySummary.TYPE_MISMATCH] *
         property_signatures.NUM_INT_COMPARISON_PROPERTIES))

    # Inputs 2 and 3 are "padding"
    start_index = (property_signatures.NUM_SINGLE_VALUE_PROPERTIES +
                   property_signatures.NUM_COMPARISON_PROPERTIES)
    self.assertEqual(
        signature[start_index : 3 * start_index],
        [PropertySummary.TYPE_MISMATCH] * (2 * start_index))

    # String properties of the output
    start_index = 3 * start_index
    self.assertEqual(signature[start_index : start_index + 5], [
        PropertySummary.ALL_FALSE,  # is empty?
        PropertySummary.ALL_FALSE,  # is single char?
        PropertySummary.MIXED,  # is short string?
        PropertySummary.MIXED,  # is lowercase?
        PropertySummary.MIXED  # is uppercase?
    ])

    # Integer properties of the output
    start_index += property_signatures.NUM_STRING_PROPERTIES
    end_index = start_index + property_signatures.NUM_INT_PROPERTIES
    self.assertEqual(
        signature[start_index : end_index],
        ([PropertySummary.TYPE_MISMATCH] *
         property_signatures.NUM_INT_PROPERTIES))

    self.assertLen(
        signature,
        (property_signatures.MAX_INPUTS + 1) *
        property_signatures.NUM_SINGLE_VALUE_PROPERTIES +
        property_signatures.MAX_INPUTS *
        property_signatures.NUM_COMPARISON_PROPERTIES)

  def test_compute_value_signature_test(self):
    value = value_module.InputValue(['butter', 'abc', 'xyz'], 'input_1')
    output = value_module.OutputValue(['butterfly', 'abc_', 'XYZ_'])

    value_signature = property_signatures.compute_value_signature(value, output)
    example_signature = property_signatures.compute_example_signature([value],
                                                                      output)

    # value_signature should a prefix of example_signature
    expected_length = (property_signatures.NUM_STRING_PROPERTIES +
                       property_signatures.NUM_INT_PROPERTIES +
                       property_signatures.NUM_BOOL_PROPERTIES +
                       property_signatures.NUM_STRING_COMPARISON_PROPERTIES +
                       property_signatures.NUM_INT_COMPARISON_PROPERTIES)
    self.assertEqual(value_signature, example_signature[: expected_length])


if __name__ == '__main__':
  absltest.main()
