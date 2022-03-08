#!/bin/bash

# Run from the root crossbeam/ directory.

results_dir=iclr2022/logic_results
if [ -d "$results_dir" ]; then
  echo "WARNING: The results directory ${results_dir} already exists. If you continue, results may be overwritten."
  while true; do
    read -p "Continue? [y/n] " yn
    case $yn in
      [Yy]* ) break;;
      [Nn]* ) echo "Exiting."; exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
else
  mkdir -p ${results_dir}
fi

# Baseline on manual tasks
python3 -m crossbeam.experiment.run_baseline_synthesizer \
  --eval_set_pkl=crossbeam/data/logic_synthesis_manual/test-tasks.pkl \
  --domain=logic \
  --timeout=30 \
  --verbose=True \
  --json_results_file=${results_dir}/baseline.logic_synthesis_manual.json

# Baseline on synthetic tasks by size
for size in $(seq 5 14) ; do
  python3 -m crossbeam.experiment.run_baseline_synthesizer \
    --eval_set_pkl=crossbeam/data/logic_by_weight/test-tasks-size-${size}.pkl \
    --domain=logic \
    --timeout=30 \
    --verbose=True \
    --json_results_file=${results_dir}/baseline.logic_by_weight.size-${size}.json
done

# CrossBeam models on manual tasks and synthetic tasks by size
beam_size=10
data_root=crossbeam/data
models_dir=trained_models/logic/
export CUDA_VISIBLE_DEVICES=0

for run in 1 2 3 4 5 ; do
  for model in sat-new-transformer sat-new-mlp ; do

    if [[ $model == *"transformer"* ]]; then
      transformer="True"
    else
      transformer="False"
    fi

    for dataset in logic_by_weight logic_synthesis_manual ; do
      python3 -m crossbeam.experiment.run_crossbeam \
          --seed=${run} \
          --domain=logic \
          --model_type=logic \
          --max_num_inputs=4 \
          --max_search_weight=20 \
          --data_folder=${data_root}/${dataset} \
          --save_dir=${models_dir} \
          --beam_size=$beam_size \
          --gpu_list=0 \
          --num_proc=1 \
          --eval_every=1 \
          --train_steps=0 \
          --do_test=True \
          --use_ur=True \
          --timeout=600 \
          --max_values_explored=50000 \
          --load_model=${model}/model-best-valid.ckpt \
          --great_transformer=${transformer} \
          --encode_weight=True \
          --json_results_file=${results_dir}/run_${run}.${model}.${dataset}.json
    done
  done
done
