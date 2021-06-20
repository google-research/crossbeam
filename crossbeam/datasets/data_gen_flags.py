# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flags for data generation."""

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('domain', 'tuple', ['tuple', 'arithmetic', 'bustle', 'logic'],
                  'task domain')
flags.DEFINE_string('output_file', None, 'data dump')
flags.DEFINE_integer('num_tasks', 1000, '# tasks')
flags.DEFINE_integer('num_searches', 100, '# searches to perform')
flags.DEFINE_integer('data_gen_timeout', 60, 'timeout per search in seconds')
flags.DEFINE_integer('num_examples', 3, '')
flags.DEFINE_integer('num_inputs', 3, '')
flags.DEFINE_integer('min_task_weight', 3, '')
flags.DEFINE_integer('max_task_weight', 10, '')
flags.DEFINE_boolean('verbose', False, 'whether to print generated tasks')
