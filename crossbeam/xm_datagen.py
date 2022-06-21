from absl import app
from absl import flags
import os
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib import framework_defaults
from xmanager.contrib import gcs


_EXP_NAME = flags.DEFINE_string(
  'exp_name', 'deepcoder', 'Name of the experiment.', short_name='n')

flags.DEFINE_integer('tout', 1800, 'timeout')
flags.DEFINE_integer('maxw', 100, 'max search weights')
flags.DEFINE_integer('maxne', 5, 'max num ex')
flags.DEFINE_integer('maxni', 3, 'max num input')
flags.DEFINE_float('skip', 0.75, 'skip')
flags.DEFINE_float('lambdaskip', 0.5, 'lambdaskip')
flags.DEFINE_integer('num_proc', 1, 'num processes')
flags.DEFINE_string('name_prefix', 'valid', 'name prefix')
flags.DEFINE_integer('num_searches', 1000, 'num searches')
flags.DEFINE_integer('num_tasks', 1000, 'num tasks per search')
flags.DEFINE_integer('min_task_weight', 3, 'min task weight')
flags.DEFINE_integer('min_num_inputs', 1, 'min num inputs')
flags.DEFINE_integer('min_num_examples', 2, 'min num ex')
flags.DEFINE_integer('shard_size', 100000, 'shard size')

flags.DEFINE_integer('num_workers', 10, 'num jobs')

FLAGS = flags.FLAGS


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with xm_abc.create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    job_requirements = xm.JobRequirements(ram=50 * xm.GiB, cpu=FLAGS.num_proc)
    executor = xm_abc.executors.Gcp(requirements=job_requirements)

    data_folder = 't-%d-maxne-%d-maxni-%d-skip-%.2f-lambdaskip-%.2f' % (
      FLAGS.tout, FLAGS.maxne, FLAGS.maxni, FLAGS.skip, FLAGS.lambdaskip
    )
    save_dir = '/gcs/xcloud-shared/hadai/data/xlambda/%s' % data_folder
    num_searches = FLAGS.num_searches // FLAGS.num_workers
    if FLAGS.num_searches % FLAGS.num_workers > 0:
      num_searches += 1
    executable_args = {
      'domain': _EXP_NAME.value,
      'output_file': '%s/%s-tasks.pkl' % (save_dir, FLAGS.name_prefix),
      'data_gen_timeout': FLAGS.tout,
      'num_tasks': FLAGS.num_tasks,
      'num_searches': num_searches,
      'min_task_weight': FLAGS.min_task_weight,
      'max_task_weight': FLAGS.maxw,
      'min_num_examples': FLAGS.min_num_examples,
      'max_num_examples': FLAGS.maxne,
      'min_num_inputs': FLAGS.min_num_inputs,
      'max_num_inputs': FLAGS.maxni,
      'skip_probability': FLAGS.skip,
      'lambda_skip_probability': FLAGS.lambdaskip,
      'choose_half_with_lambdas': True,
      'num_datagen_proc': FLAGS.num_proc,
      'shard_size': FLAGS.shard_size,
      'verbose': False
    }
    module = 'crossbeam.datasets.bottom_up_data_generation'
    executable, = experiment.package([
      xm.python_container(
        path='.',
        base_image=framework_defaults.base_image(
          'pytorch', job_requirements.accelerator),
        entrypoint=xm.ModuleName(module),
        use_deep_module=True,
        executor_spec=executor.Spec(),
        args=executable_args)
        ])
    job = xm.Job(executable, executor)
    nshard_per_job = num_searches * FLAGS.num_tasks // (FLAGS.num_workers * FLAGS.shard_size) + 1
    nshard_per_job = max(nshard_per_job, num_searches)
    job_configs = list([{'data_gen_seed': x * nshard_per_job} for x in range(FLAGS.num_workers)])

    for job_args in job_configs:
      experiment.add(job, args={'args': job_args})


if __name__ == '__main__':
  app.run(main)
