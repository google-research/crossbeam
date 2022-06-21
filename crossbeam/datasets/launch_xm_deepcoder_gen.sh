#!/bin/bash

tout=1800
maxw=100  # Run until timeout.
maxne=5
maxni=3
skip=0.75
lambdaskip=0.5
num_proc=4
num_workers=100
phase=valid

xmanager launch \
  xm_datagen.py -- \
  --xm_resource_alloc="user:xcloud/${USER}" \
  --xm_gcs_path=/gcs/xcloud-shared/${USER}/xlambda \
  --name_prefix=$phase \
  --tout=$tout \
  --num_tasks=1000 \
  --num_searches=1000 \
  --num_workers=$num_workers \
  --min_task_weight=3 \
  --maxw=$maxw \
  --min_num_examples=2 \
  --maxne=$maxne \
  --min_num_inputs=1 \
  --maxni=$maxni \
  --skip=$skip \
  --lambdaskip=$lambdaskip \
  --num_proc=$num_proc
