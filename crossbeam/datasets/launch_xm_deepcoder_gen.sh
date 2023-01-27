#!/bin/bash

tout=3600
split=valid
num_tasks_per_weight=5
num_searches=1000  # Total across all workers
num_workers=50
num_proc=4
lambda_fraction=0.8
shuffle_ops=False
start_seed=0

xmanager launch \
  xm_datagen.py -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc="user:xcloud/${USER}" \
  --xm_gcs_path=/gcs/xcloud-shared/${USER}/xlambda \
  --user=${USER} \
  --exp_name=gen-deepcoder-${split}-data \
  --domain=deepcoder \
  --tout=$tout \
  --split=$split \
  --num_tasks_per_weight=$num_tasks_per_weight \
  --num_searches=$num_searches \
  --num_workers=$num_workers \
  --num_proc=$num_proc \
  --min_task_weight=3 \
  --max_task_weight=15 \
  --min_num_examples=2 \
  --max_num_examples=5 \
  --min_num_inputs=1 \
  --max_num_inputs=3 \
  --skip=0.0 \
  --lambdaskip=0.0 \
  --lambda_fraction=$lambda_fraction \
  --shuffle_ops=$shuffle_ops \
  --start_seed=$start_seed \


split=train
num_tasks_per_weight=200
num_searches=10000  # Total across all workers
num_workers=500
num_proc=4
start_seed=10000

xmanager launch \
  xm_datagen.py -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc="group:xcloud/xcloud-shared-user" \
  --xm_gcs_path=/gcs/xcloud-shared/${USER}/xlambda \
  --user=${USER} \
  --exp_name=gen-deepcoder-${split}-data \
  --domain=deepcoder \
  --tout=$tout \
  --split=$split \
  --num_tasks_per_weight=$num_tasks_per_weight \
  --num_searches=$num_searches \
  --num_workers=$num_workers \
  --num_proc=$num_proc \
  --min_task_weight=3 \
  --max_task_weight=15 \
  --min_num_examples=2 \
  --max_num_examples=5 \
  --min_num_inputs=1 \
  --max_num_inputs=3 \
  --skip=0.0 \
  --lambdaskip=0.0 \
  --lambda_fraction=$lambda_fraction \
  --shuffle_ops=$shuffle_ops \
  --start_seed=$start_seed \
