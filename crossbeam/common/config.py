from absl import flags

"""
We define a set of args here to avoid repetitions in each main function.

The set of args included here should be consistent in different scenarios and should have no ambiguity.
"""

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string('data_folder', None, 'folder for offline data dump')
flags.DEFINE_string('save_dir', None, 'folder for saving model dump/logs')

# nn configs
flags.DEFINE_integer('batch_size', 32, 'minibatch size')
flags.DEFINE_integer('embed_dim', 128, 'embedding dimension')
flags.DEFINE_integer('n_para_dataload', 0, 'num of parallel data loader')
flags.DEFINE_integer('decoder_rnn_layers', 3, '# rnn autoregressive model decoder layers')
flags.DEFINE_float('grad_clip', 5.0, 'clip grad')
flags.DEFINE_integer('train_steps', 10000, 'number of training steps')
flags.DEFINE_integer('eval_every', 1000, 'number of steps between evals')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_integer('gpu', -1, '')


flags.DEFINE_integer('beam_size', 4, '')