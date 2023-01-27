from copy import deepcopy
import getpass
import os
import subprocess

from absl import app
from absl import flags
from ml_collections import config_flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib import framework_defaults
from xmanager.contrib import gcs
from xmanager.contrib.internal import tensorboard


_EXP_NAME = flags.DEFINE_string(
  'exp_name', 'deepcoder', 'Name of the experiment.', short_name='n')

config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string='Training configuration file.',
    lock_config=True)

flags.DEFINE_string('save_folder_pattern',
                    '/gcs/xcloud-shared/{user}/results/xlambda/{exp_name}_{exp_id}',
                    'save folder pattern')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus per job')
FLAGS = flags.FLAGS


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  job_config = FLAGS.config

  config_filename = config_flags.get_config_filename(FLAGS['config'])
  config_file_name = os.path.join('/workdir/crossbeam', config_filename)
  executable_args = {}
  # Add config flag and related overrides to args.
  executable_args['config'] = config_file_name
  # Capture all the flag values passed in command line and relay them to binary.
  executable_args.update({
      name: value
      for name, value in FLAGS.flag_values_dict().items()
      if name.startswith('config.')
  })
  uname = getpass.getuser()

  with xm_abc.create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    job_requirements = xm.JobRequirements(ram=8 * FLAGS.num_gpus * xm.GiB,
                                          cpu=4 * FLAGS.num_gpus,
                                          v100=FLAGS.num_gpus)
    executor = xm_abc.executors.Gcp(requirements=job_requirements)
    save_dir = FLAGS.save_folder_pattern.format(
        user=uname, exp_name=_EXP_NAME.value, exp_id=experiment.experiment_id)

    executable_args.update({
        # Hanjun has the "master copy" of the data.
        'config.data_root': '/gcs/xcloud-shared/hadai/data/xlambda',
    })
    module = 'crossbeam.experiment.run_crossbeam'
    executable, = experiment.package([
      xm.python_container(
        path='.',
        base_image='gcr.io/deeplearning-platform-release/pytorch-gpu.1-12',
        entrypoint=xm.ModuleName(module),
        use_deep_module=True,
        executor_spec=executor.Spec(),
        args=executable_args)
        ])

    async def make_job(work_unit, **kwargs):
      args = deepcopy(executable_args)
      sweep_str_parts = []
      for k, v in kwargs.items():
        if k.startswith('config.'):
          k = k[len('config.'):]
        sweep_str_parts.append(f'{k}={v!r}')
      sweep_str = ','.join(sweep_str_parts)
      args['config.save_dir'] = f'{save_dir}/{work_unit.work_unit_id}_{sweep_str}'
      args.update(kwargs)
      work_unit.add(xm.Job(executable, args=args, executor=executor))

    for sweep_args in job_config.get('sweep', [{}]):
      experiment.add(make_job, args=sweep_args)

    tensorboard.add_tensorboard_corp(experiment, save_dir)
    tensorboard.add_tensorboard_borg(experiment, save_dir)

    config_save_location = os.path.join(
        save_dir.replace('/gcs/', 'gs://'),
        'config',
        os.path.basename(config_filename))
    copy_config_command = ['gsutil', 'cp', config_filename,
                           config_save_location]
    print(f'Executing command: {" ".join(copy_config_command)}')
    subprocess.run(copy_config_command, check=True)

if __name__ == '__main__':
  app.run(main)
