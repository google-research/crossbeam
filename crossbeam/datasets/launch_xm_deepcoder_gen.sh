#!/bin/bash

tout=3600
maxw=15  # Run until timeout.
maxne=5
maxni=3
skip=0.0
lambdaskip=0.0
lambda_fraction=0.8
shuffle_ops=False
num_proc=4
num_workers=2000
split=train
start_seed=1000000

xmanager launch \
  xm_datagen.py -- \
  --xm_resource_alloc="user:xcloud/${USER}" \
  --xm_gcs_path=/gcs/xcloud-shared/${USER}/xlambda \
  --user=${USER} \
  --split=$split \
  --tout=$tout \
  --num_tasks_per_weight=200 \
  --num_searches=10000 \
  --num_workers=$num_workers \
  --min_task_weight=3 \
  --maxw=$maxw \
  --min_num_examples=2 \
  --maxne=$maxne \
  --min_num_inputs=1 \
  --maxni=$maxni \
  --skip=$skip \
  --lambdaskip=$lambdaskip \
  --lambda_fraction=$lambda_fraction \
  --shuffle_ops=$shuffle_ops \
  --start_seed=$start_seed \
  --num_proc=$num_proc
