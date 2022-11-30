from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  config.sweep = [{'config.lr': 1e-4}, {'config.lr': 1e-3}]

  config.tout = 3600
  config.domain = 'deepcoder'
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.min_task_weight = 3
  config.max_task_weight = 14
  config.min_num_examples = 2
  config.max_num_examples = 5
  config.min_num_inputs = 1
  config.max_num_inputs = 3
  config.max_search_weight = 12
  config.grad_accumulate = 4
  config.beam_size = 10
  config.gpu_list = '0,1,2,3,4,5,6,7'
  config.num_proc = 8
  config.embed_dim = 64
  config.eval_every = 5000
  config.num_valid = 250
  config.use_ur = False
  config.encode_weight = True
  config.train_steps = 1000000
  config.train_data_glob = 'train-tasks*.pkl'
  config.random_beam = False
  config.lr = 1e-4
  config.data_name = 't-3600-maxne-5-maxni-3-skip-0.00-lambdaskip-0.00-lambdafrac-0.80'
  return config
