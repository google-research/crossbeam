## Setup xcloud


## Run experiments locally

First you need to define the experimental config file, see examples under `experiment/deepcoder/configs`

Then you can specify the config file and run on deepcoder as:

```
cd experiment/deepcoder
config_name=exp_singlegpu ./run_deepcoder.sh
```

Or, using multiple GPUs:

```
cd experiment/deepcoder
config_name=exp_multigpu devices=0,1,2,3,4,5,6,7 ./run_deepcoder.sh
```

## Launch experiments on xcloud

Navigate to the the folder under `crossbeam/`, and then run:

```
config=experiment/deepcoder/configs/exp_singlegpu.py name=your_exp_name ./experiment/launch_xm_train.sh --num_gpus=1
```

