# CrossBeam: Learning to Search in Bottom-Up Program Synthesis

This repository contains the source code associated with the paper published at
ICLR'22:

> CrossBeam: Learning to Search in Bottom-Up Program Synthesis

In this research project, we train a neural model to learn a hands-on search
policy for bottom-up program synthesis, in an effort to tame the search space
blowup.

TODO(kshi): Link to arxiv and/or OpenReview here

## Setup

For dependencies, first install
[PyTorch](https://pytorch.org/get-started/locally/) with CUDA and
`torch-scatter`, for example with the following commands:

```
pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

Then run the following from the root `crossbeam/` directory:

```
pip install -e .
```

You can also append the `--user` option if you don't want to install globally.

## Running tests

From the root `crossbeam` directory:

```
pytest
```

## Generate training and test datasets

The synthetic training data is only needed for re-training models and is not
included in this repo. The following steps will generate the synthetic training
data:

```
./crossbeam/datasets/run_bustle_gen.sh
./crossbeam/datasets/run_logic_gen.sh
./crossbeam/datasets/run_logic_clean_and_shard.sh
```

Test datasets are in `crossbeam/data/`.

## Train the model

Trained models are included in this repo in `trained_models/`. If you wish to
re-train the models, first generate training data with the commands above, and
then run:

```
TODO(hadai)
```

## Running the trained model on test datasets

From the root `crossbeam/` directory, run:

```
./iclr2022/bustle_experiments.sh
./iclr2022/logic_experiments.sh
```

These commands will save results as JSON files in `iclr2022/bustle_results` and
`iclr2022/logic_results`.

## Disclaimer

This is not an official Google product.
