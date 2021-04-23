from absl import flags

"""
We define a set of args here to avoid repetitions in each main function.

The set of args included here should be consistent in different scenarios and should have no ambiguity.
"""

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string('data_folder', None, 'folder for offline data dump')
