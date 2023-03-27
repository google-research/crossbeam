#!/bin/bash

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


save_dir=${save_dir?}
config=${config?}
name=${name?}
timeout=${timeout?=60}
run=${run:=0}
runs=${runs:=1}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=${devices:=0}

limit=$((runs + run - 1))
for i in $(seq ${run} ${limit}); do
    python3 -m crossbeam.experiment.run_crossbeam \
        --config="${config}" \
        --config.save_dir=${save_dir} \
        --config.data_root="${HOME}/xlambda-data/deepcoder" \
        --config.do_test=True \
        --config.timeout=${timeout} \
        --config.num_proc=1 \
        --config.gpu_list=0 \
        --config.gpu=0 \
        --config.port='29501' \
        --config.seed=${i} \
        --config.train_data_glob='' \
        --config.test_data_glob='' \
        --config.json_results_file=$save_dir/results.${name}.timeout${timeout}.run${i}.json \
        --config.load_model=${save_dir}/model-best-valid.ckpt \
        $@
done

#        --config.use_ur=True \
