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

export CUDA_VISIBLE_DEVICES=0

# Model settings.
model_dir=$HOME/xlambda-results/deepcoder/nov07-lr-4e-4-gpus-8-grad_acc-4

io=lambda_signature
value=lambda_signature

# Evaluation settings.
beam_size=10
timeout=60
max_values_explored=999999999
maxni=3
maxsw=12

python -m crossbeam.experiment.run_crossbeam \
    --domain=deepcoder \
    --model_type=deepcoder \
    --io_encoder=${io} \
    --value_encoder=${value} \
    --max_num_inputs=$maxni \
    --encode_weight=True \
    --embed_dim=64 \
    --num_proc=1 \
    --gpu_list=0 \
    --do_test=True \
    --train_steps 0 \
    --eval_every 1 \
    --num_valid 0 \
    --save_dir='' \
    --load_model=${model_dir}/model-best-valid.ckpt \
    --timeout=${timeout} \
    --max_values_explored=${max_values_explored} \
    --max_search_weight=$maxsw \
    --beam_size $beam_size \
    --use_ur=True \
    --stochastic_beam=False \
    --json_results_file=${model_dir}/results.json \
    $@

