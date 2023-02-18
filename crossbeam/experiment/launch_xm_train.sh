#!/bin/bash


xmanager launch \
  xm_train.py -- \
  --xm_resource_alloc=group:xcloud/xcloud-shared-user \
  --xm_gcs_path=/gcs/xcloud-shared/${USER}/xlambda \
  --config=${config?} \
  --exp_name=${name?} \
  $@
