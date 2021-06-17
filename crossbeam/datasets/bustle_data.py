import random


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


# TODO(kshi): Reimplement GenerateData::getRandomExamples


def bustle_constants_extractor(task):
  # TODO(kshi): Reimplement ConstantExtraction::extractConstants
  del task
  return ['', ' ', 0, 1, 2, 99]

