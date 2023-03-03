from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  config.sweep = [{'config.schedule_type': 'halfhalf-2'},
                  {'config.schedule_type': 'uniform'},
                  {'config.schedule_type': 'all-3'},
                  {'config.schedule_type': 'all-4'},
                  {'config.schedule_type': 'all-5'},
                  {'config.schedule_type': 'all-6'}]

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
  config.num_proc = 8
  config.gpu_list = '0,1,2,3,4,5,6,7'
  config.gpu = 0
  config.embed_dim = 64
  config.eval_every = 5000
  config.num_valid = 250
  config.port = '29500'
  config.use_ur = False
  config.do_test = False
  config.json_results_file = ''
  config.timeout = 60
  config.encode_weight = True
  config.train_steps = 1000000
  config.train_data_glob = 'train-*.pkl'
  config.test_data_glob = 'valid-*.pkl'
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.lr = 5e-4
  config.load_model = ''
  config.steps_per_curr_stage = 5000
  config.schedule_type = 'halfhalf'
  config.data_name = 't-3600-lambdafrac-0.8-shuffleops-False'
  return config
