#!/bin/bash

python3 -m crossbeam.experiment.run_baseline_synthesizer \
  --eval_set_pkl=~/data/crossbeam/logic_synthesis_manual/test-tasks.pkl \
  --domain=logic \
  --timeout=30 \
  --json_results_file=results/baseline.logic-handcrafted.30s.json \
  $@
  #--max_values_explored=None \
