import unique_randomizer as ur

randomizer = ur.UniqueRandomizer()

distributions = {
    '': [0.2, 0.1, 0.6],
    '0': [0.1, 0.1, 0.4],
    '1': [0.3, 0.7, 0.2],
    '2': [0.2, 0.5, 0.1],
}

for i in range(8):
  seq = ''
  for _ in range(2):
    choice = randomizer.sample_distribution([0.5, 0.2, 0.1])
    seq += str(choice)
  randomizer.mark_sequence_complete()

  print('Sample {}: {}'.format(i, seq))

print('\nAdding more options!\n')

distributions = {
    '': [0.2, 0.1, 0.6, 0.2, 0.1],
    '0': [0.1, 0.1, 0.4, 0.7, 0.3],
    '1': [0.3, 0.7, 0.2, 0.1, 0.1],
    '2': [0.2, 0.5, 0.1, 0.3, 0.4],
    '3': [0.1, 0.4, 0.2, 0.3, 0.1],
    '4': [0.2, 0.1, 0.2, 0.7, 0.1],
}


for i in range(17):
  seq = ''
  for _ in range(2):
    choice = randomizer.sample_distribution([0.5, 0.2, 0.1, 0.7, 0.2])
    seq += str(choice)
  randomizer.mark_sequence_complete()

  print('Sample {}: {}'.format(i, seq))
