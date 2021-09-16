"""Property signatures for Crossbeam."""

import enum
import re

MAX_INPUTS = 3

NUM_STRING_PROPERTIES = 14
NUM_INT_PROPERTIES = 7
NUM_BOOL_PROPERTIES = 1
NUM_STRING_COMPARISON_PROPERTIES = 17
NUM_INT_COMPARISON_PROPERTIES = 7
NUM_SINGLE_VALUE_PROPERTIES = (
    NUM_STRING_PROPERTIES + NUM_INT_PROPERTIES + NUM_BOOL_PROPERTIES)
NUM_COMPARISON_PROPERTIES = (
    NUM_STRING_COMPARISON_PROPERTIES + NUM_INT_COMPARISON_PROPERTIES)

_CONTAINS_DIGITS_REGEX = re.compile(r'.*\d.*')
_ONLY_DIGITS_REGEX = re.compile(r'\d+')
_CONTAINS_LETTERS_REGEX = re.compile(r'.*[a-zA-Z].*')
_ONLY_LETTERS_REGEX = re.compile(r'[a-zA-Z]+')

@enum.unique
class PropertySummary(enum.IntEnum):
  """Enum summarizing the result of a property function."""
  # Padding, should be ignored.
  PADDING = 0
  # The property returns true for all inputs.
  ALL_TRUE = 1
  # The property returns false for all inputs.
  ALL_FALSE = 2
  # The property returns true for some inputs and false for others.
  MIXED = 3
  # The property could not be evaluated for the inputs due to type mismatch.
  TYPE_MISMATCH = 4


def compute_example_signature(inputs, output):
  """Computes property signature for an example."""
  # In the BUSTLE domain, the output must be a string.
  assert output.type == str

  # Signature will look like:
  # single_value(in_1) | comparison(in_1, out) | single_value(in_2) |
  # comparison(in_2, out) | ... | singlevalue(out)
  padded_inputs = inputs + [None] * (MAX_INPUTS - len(inputs))
  signature = []
  for inp in padded_inputs:
    process_single_value(inp, signature)
    process_comparison(inp, output, signature)
  process_single_value(output, signature)
  return signature


def compute_value_signature(intermediate, output):
  """Computes property signature for a single intermediate value."""
  # In the BUSTLE domain, the output must be a string.
  assert output.type == str

  # Signature will look like:
  # single_value(intermediate) | comparison(intermediate, output)
  signature = []
  process_single_value(intermediate, signature)
  process_comparison(intermediate, output, signature)
  return signature


def process_single_value(value, signature):
  """Processes the single value and adds results to the signature."""
  value_type = value.type if value is not None else None

  if value_type == str:
    property_results = []
    for s in value.values:
      lower = s.lower()
      upper = s.upper()
      length = len(s)
      properties = [
          length == 0,  # is empty?
          length == 1,  # is single char?
          length <= 5,  # is short string?
          s == lower,  # is lowercase?
          s == upper,  # is uppercase?
          ' ' in s,  # contains space?
          ',' in s,  # contains comma?
          '.' in s,  # contains period?
          '-' in s,  # contains dash?
          '/' in s,  # contains slash?
          _CONTAINS_DIGITS_REGEX.match(s),  # contains digits?
          _ONLY_DIGITS_REGEX.match(s),  # only digits?
          _CONTAINS_LETTERS_REGEX.match(s),  # contains letters?
          _ONLY_LETTERS_REGEX.match(s),  # only letters?
      ]
      assert len(properties) == NUM_STRING_PROPERTIES
      property_results.append(properties)
    reduce_property_booleans(property_results, signature)
  else:
    signature.extend([PropertySummary.TYPE_MISMATCH] * NUM_STRING_PROPERTIES)

  if value_type == int:
    property_results = []
    for integer in value.values:
      properties = [
          integer == 0,  # is zero?
          integer == 1,  # is one?
          integer == 2,  # is two?
          integer < 0,  # is negative?
          0 < integer <= 3,  # is small integer?
          3 < integer <= 9,  # is medium integer?
          9 < integer,  # is large integer?
      ]
      assert len(properties) == NUM_INT_PROPERTIES
      property_results.append(properties)
    reduce_property_booleans(property_results, signature)
  else:
    signature.extend([PropertySummary.TYPE_MISMATCH] * NUM_INT_PROPERTIES)

  if value_type == bool:
    property_results = []
    for boolean in value.values:
      properties = [
          boolean,  # is the boolean true?
      ]
      assert len(properties) == NUM_BOOL_PROPERTIES
      property_results.append(properties)
    reduce_property_booleans(property_results, signature)
  else:
    signature.extend([PropertySummary.TYPE_MISMATCH] * NUM_BOOL_PROPERTIES)


def process_comparison(value, output, signature):
  """Processes a comparison between two values, adding results to signature."""
  value_type = value.type if value is not None else None

  if value_type == str:
    property_results = []
    for s, output_s in zip(value.values, output.values):
      lower = s.lower()
      output_s_lower = output_s.lower()
      properties = [
          s in output_s,  # output contains input?
          output_s.startswith(s),  # output starts with input?
          output_s.endswith(s),  # output ends with input?
          output_s in s,  # input contains output?
          s.startswith(output_s),  # input starts with output?
          s.endswith(output_s),  # input ends with output?
          lower in output_s_lower,  # output contains input ignoring case?
          output_s_lower.startswith(lower),  # out starts with in ignoring case?
          output_s_lower.endswith(lower),  # out ends with in ignoring case?
          output_s_lower in lower,  # input contains output ignoring case?
          lower.startswith(output_s_lower),  # in starts with out ignoring case?
          lower.endswith(output_s_lower),  # in ends with out ignoring case?
          s == output_s,  # input equals output?
          lower == output_s_lower,  # input equals output ignoring case?
          len(s) == len(output_s),  # input same length as output?
          len(s) < len(output_s),  # input shorter than output?
          len(s) > len(output_s),  # input longer than output?
      ]
      assert len(properties) == NUM_STRING_COMPARISON_PROPERTIES
      property_results.append(properties)
    reduce_property_booleans(property_results, signature)
  else:
    signature.extend([PropertySummary.TYPE_MISMATCH] *
                     NUM_STRING_COMPARISON_PROPERTIES)

  if value_type == int:
    property_results = []
    for integer, output_s in zip(value.values, output.values):
      len_output_s = len(output_s)
      properties = [
          integer < len_output_s,  # is less than output length?
          integer <= len_output_s,  # is less or equal to output length?
          integer == len_output_s,  # is equal to output length?
          integer >= len_output_s,  # is greater or equal to output length?
          integer > len_output_s,  # is greater than output length?
          abs(integer - len_output_s) <= 1,  # is very close to output length?
          abs(integer - len_output_s) <= 3,  # is close to output length?
      ]
      assert len(properties) == NUM_INT_COMPARISON_PROPERTIES
      property_results.append(properties)
    reduce_property_booleans(property_results, signature)
  else:
    signature.extend([PropertySummary.TYPE_MISMATCH] *
                     NUM_INT_COMPARISON_PROPERTIES)


def reduce_property_booleans(property_results, signature):
  """Reduces property booleans across examples."""
  transposed = zip(*property_results)
  for property_across_examples in transposed:
    has_true = any(property_across_examples)
    has_false = not all(property_across_examples)
    if has_true and has_false:
      signature.append(PropertySummary.MIXED)
    elif has_true and not has_false:
      signature.append(PropertySummary.ALL_TRUE)
    elif not has_true and has_false:
      signature.append(PropertySummary.ALL_FALSE)
    else:
      raise Exception('single_example_results contained neither True nor False')
