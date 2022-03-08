#!/bin/bash

# Run from the root crossbeam/ directory.

results_dir=iclr2022/bustle_results
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

# Baseline
python3 -m crossbeam.experiment.run_baseline_synthesizer \
  --eval_set_pkl=crossbeam/data/sygus/test-tasks-sygus.pkl \
  --domain=bustle \
  --timeout=30 \
  --verbose=True \
  --json_results_file=${results_dir}/baseline.sygus.30s.json

python3 -m crossbeam.experiment.run_baseline_synthesizer \
  --eval_set_pkl=crossbeam/data/new/test-tasks-new.pkl \
  --domain=bustle \
  --timeout=30 \
  --verbose=True \
  --json_results_file=${results_dir}/baseline.new.30s.json

# CrossBeam
maxni=3
maxsw=20
beam_size=10
data_root=crossbeam/data
models_dir=trained_models/bustle/
export CUDA_VISIBLE_DEVICES=0

for run in 1 2 3 4 5 ; do
  for model in vw-bustle_sig-vsize randbeam ; do
    for dataset in sygus new ; do
      # Normal CrossBeam with UR for evaluation
      python3 -m crossbeam.experiment.run_crossbeam \
          --seed=${run} \
          --domain=bustle \
          --model_type=char \
          --max_num_inputs=$maxni \
          --max_search_weight=$maxsw \
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
          --io_encoder=bustle_sig --value_encoder=bustle_sig --encode_weight=True \
          --json_results_file=${results_dir}/run_${run}.${model}.${dataset}.json

      if [[ "${model}" != "randbeam" ]] ; then
        if (( $run == 1)) ; then
          # CrossBeam with Beam Search during evaluation
          # This is deterministic, only 1 run needed
          python3 -m crossbeam.experiment.run_crossbeam \
              --seed=${run} \
              --domain=bustle \
              --model_type=char \
              --max_num_inputs=$maxni \
              --max_search_weight=$maxsw \
              --data_folder=${data_root}/${dataset} \
              --save_dir=${models_dir} \
              --beam_size=$beam_size \
              --gpu_list=0 \
              --num_proc=1 \
              --eval_every=1 \
              --train_steps=0 \
              --do_test=True \
              --use_ur=False \
              --timeout=600 \
              --max_values_explored=50000 \
              --load_model=${model}/model-best-valid.ckpt \
              --io_encoder=bustle_sig --value_encoder=bustle_sig --encode_weight=True \
              --json_results_file=${results_dir}/run_${run}.${model}.beam_search.${dataset}.json
        fi

        # CrossBeam with multinomial random sampling during evaluation
        python3 -m crossbeam.experiment.run_crossbeam \
            --seed=${run} \
            --domain=bustle \
            --model_type=char \
            --max_num_inputs=$maxni \
            --max_search_weight=$maxsw \
            --data_folder=${data_root}/${dataset} \
            --save_dir=${models_dir} \
            --beam_size=$beam_size \
            --gpu_list=0 \
            --num_proc=1 \
            --eval_every=1 \
            --train_steps=0 \
            --do_test=True \
            --use_ur=False \
            --stochastic_beam=True \
            --timeout=600 \
            --max_values_explored=50000 \
            --load_model=${model}/model-best-valid.ckpt \
            --io_encoder=bustle_sig --value_encoder=bustle_sig --encode_weight=True \
            --json_results_file=${results_dir}/run_${run}.${model}.stochastic_beam.${dataset}.json
      fi
    done
  done
done
