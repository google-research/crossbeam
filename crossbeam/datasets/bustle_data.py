import itertools
import random

from crossbeam.datasets import random_data

CHARSETS = [
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Uppercase letters
    'abcdefghijklmnopqrstuvwxyz',  # Lowercase letters
    '0123456789',  # Digits
    ' ',  # Space
    '.,-+_@$/',  # Common punctuation
]


def bustle_input_generator():
  # GenerateData::randomInput
  length = random.randint(1, 10)
  usable_charsets = [charset for charset in CHARSETS
                     if random.random() < 0.25]
  if not usable_charsets:
    usable_charsets.append(CHARSETS[1])  # Lowercase letters
  return ''.join(random.choice(random.choice(usable_charsets))
                 for _ in range(length))


bustle_inputs_dict_generator = random_data.make_inputs_dict_generator(
    bustle_input_generator)

# TODO(kshi): Reimplement GenerateData::getRandomExamples

ALWAYS_USED_CONSTANTS = ['', 0, 1, 2, 3, 99]
COMMON_CONSTANTS = [
    ' ', ',', '.', '!', '?', '(', ')', '[', ']', '<', '>', '{', '}', '-', '+',
    '_', '/', '$', '#', ':', ';', '@', '%', '0']


def compute_lcs(str1, str2):
  len1 = len(str1)
  len2 = len(str2)
  # dp[i][j] = length of longest common substring of str1[:i] and str2[:j]
  dp = [[0] * (len2+1) for _ in range(len1+1)]

  bestSubstring = ''
  bestLength = 0
  for i in range(1, len1 + 1):
    for j in range(1, len2 + 1):
      if str1[i-1] == str2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
        if dp[i][j] > bestLength:
          bestLength = dp[i][j]
          bestSubstring = str1[i - bestLength : i]
  return bestSubstring


def bustle_constants_extractor(task):
  """Extracts constants from the input-output pairs.

  Extracted constants include:
   1. The longest common susbtring between any pair of output strings, or any
      pair of input strings from the same column, if it has >= 2 characters.
   2. Any string constant from the provided list of common string constants, if
      it is a substring of any input or output.
  """
  # TODO(kshi): Reimplement ConstantExtraction::extractConstants

  extracted_constants = []
  extracted_constants_set = set()

  string_columns = list(task.inputs_dict.values())
  if all(isinstance(output, str) for output in task.outputs):
    string_columns.append(task.outputs)

  # Criteria 1: longest common substrings.
  lcs_pairs = sum((list(itertools.combinations(column, 2))
                   for column in string_columns), [])
  for x, y in lcs_pairs:
    lcs = compute_lcs(x, y)
    if len(lcs) >= 2:
      extracted_constants_set.add(lcs)

  # Criteria 2: common string constants.
  for common in COMMON_CONSTANTS:
    if any(common in s for column in string_columns for s in column):
      extracted_constants_set.add(common)

  # Sort the constants by length and then alphabetically.
  extracted_constants = list(extracted_constants_set)
  sorted(extracted_constants_set, key=lambda s: (len(s), s))

  return ALWAYS_USED_CONSTANTS + extracted_constants
