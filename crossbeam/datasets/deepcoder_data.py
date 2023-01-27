"""Creates random input data for DeepCoder problems."""

import itertools
import random

# Ranges have inclusive endpoints and are paired with sampling weights.
INT_RANGES = [
    # Positive and negative ranges.
    ((-256, 256), 10),
    ((-100, 100), 8),
    ((-50, 50), 7),
    ((-10, 10), 6),
    # Nonnegative ranges.
    ((0, 256), 5),
    ((0, 100), 4),
    ((0, 50), 3),
    ((0, 10), 5),
    # Nonpositive ranges.
    ((-256, 0), 2),
    ((-50, 0), 2),
    ((-10, 0), 2),
    # Specific numbers of digits.
    ((100, 256), 2),
    ((10, 99), 4),
    ((1, 9), 4),
    ((-9, -1), 3),
    ((-99, -10), 2),
    # Negative numbers as sentinel values.
    ((-5, 50), 3),
    ((-1, 10), 2),
    ((-1, 5), 2),
    # Binary and ternary.
    ((0, 1), 4),
    ((-1, 1), 3),
    ((0, 2), 2),
    # Potential indices.
    ((0, 9), 5),
    # Constant.
    ((0, 0), 1),
    ((-1, -1), 1),
    ((1, 1), 1),
]
_INT_RANGES_WEIGHTS = [weight for _, weight in INT_RANGES]


LENGTH_RANGES = [
    ((1, 10), 5),
    ((1, 5), 4),
    ((4, 7), 3),
    ((6, 10), 2),
    ((3, 5), 3),
    ((1, 1), 1),
    ((2, 2), 2),
    ((3, 3), 1),
    ((4, 4), 1),
]
_LENGTH_RANGES_WEIGHTS = [weight for _, weight in LENGTH_RANGES]


def flip(true_probability=0.5):
  return random.random() < true_probability


def random_int(input_format):
  min_int, max_int = input_format['int_range']
  return random.randint(min_int, max_int)


def random_int_list(input_format):
  """Samples a random int list according to some list settings."""
  min_length, max_length = input_format['length_range']
  list_length = random.randint(min_length, max_length)

  min_int, max_int = input_format['int_range']
  element_range = range(min_int, max_int + 1)
  if input_format['unique'] and max_length <= len(element_range):
    the_list = random.sample(element_range, k=list_length)
  else:
    the_list = random.choices(element_range, k=list_length)

  if input_format['sort'] == 'inc':
    the_list = sorted(the_list)
  elif input_format['sort'] == 'dec':
    the_list = sorted(the_list, reverse=True)
  else:
    assert input_format['sort'] == 'none'

  return the_list


def deepcoder_inputs_dict_generator(num_inputs, num_examples):
  """Returns a dict of random inputs for DeepCoder."""
  while True:  # Try until there are no identical inputs or examples.
    inputs_dict = {}
    input_names = []
    input_formats = []
    input_types = []

    for i in range(num_inputs):
      input_name = f'x{i + 1}'
      input_type = random.choice(['int', 'int_list', 'int_list'])

      copy_list_lengths_index = -1

      if i > 0 and flip():
        # Copy the format of some previous input. The input type might be
        # different, but the range of ints would be the same.
        index_to_copy = random.randrange(i)
        input_format = input_formats[index_to_copy]
        if (input_type == 'int_list'
            and input_types[index_to_copy] == 'int_list' and flip(0.75)):
          copy_list_lengths_index = index_to_copy
      else:
        input_format = {}

      if 'int_range' not in input_format:
        int_range, _ = random.choices(INT_RANGES, weights=_INT_RANGES_WEIGHTS,
                                      k=1)[0]
        input_format['int_range'] = int_range

      if input_type == 'int':
        input_creator_fn = random_int

      elif input_type == 'int_list':
        input_creator_fn = random_int_list
        if 'length_range' not in input_format:
          length_range, _ = random.choices(
              LENGTH_RANGES, weights=_LENGTH_RANGES_WEIGHTS, k=1)[0]
          input_format['length_range'] = length_range
          input_format['unique'] = flip(0.25)
          input_format['sort'] = random.choices(['none', 'inc', 'dec'],
                                                weights=[7, 2, 1], k=1)[0]

      else:
        raise ValueError(f'Unhandled input_type: {input_type}')

      if copy_list_lengths_index != -1:
        input_examples = []
        for other_list in inputs_dict[input_names[copy_list_lengths_index]]:
          specific_len_input_format = input_format.copy()
          list_len = len(other_list)
          specific_len_input_format['length_range'] = (list_len, list_len)
          input_examples.append(input_creator_fn(specific_len_input_format))
      else:
        input_examples = [input_creator_fn(input_format)
                          for _ in range(num_examples)]
      inputs_dict[input_name] = input_examples

      input_names.append(input_name)
      input_formats.append(input_format)
      input_types.append(input_type)

    ok = True
    for name_1, name_2 in itertools.combinations(input_names, 2):
      if inputs_dict[name_1] == inputs_dict[name_2]:
        ok = False
    for example_1, example_2 in itertools.combinations(range(num_examples), 2):
      if all(inputs_dict[name][example_1] == inputs_dict[name][example_2]
             for name in input_names):
        ok = False
    if ok:
      return inputs_dict
