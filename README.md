# CrossBeam: Learned beam search

This repository contains the source code associated with the paper published at
ICLR'22:

> CrossBeam: Learning to Search in Bottom-Up Program Synthesis

In this research project, we propose training a neural model to learn a
hands-on search policy for bottom-up program synthesis, in an effort to tame
the search space blowup.

TODO(kshi): Link to arxiv and/or OpenReview here

## Setup

We use pip package for management. Please first do pip install at the root
folder:

    pip install -e .

You can also append `--user` option if you don't want to install globally.

## Running tests

From the root `crossbeam` directory:

```
pytest
```

## Generate training and test datasets

Training data is not included in this repo, but we include steps to generate the
synthetic training data:
```
TODO(kshi)
```

Test datasets are in `crossbeam/data/`, but they can also be generated:
```
TODO(kshi)
```

## Train the model

```
TODO(hadai)
```

## Running the trained model on test datasets

From the root `crossbeam/` directory:
```
./iclr2022/bustle_experiments.sh
./iclr2022/logic_experiments.sh
```

## Disclaimer

This is not an official Google product.
