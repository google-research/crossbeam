# CrossBeam: Learned beam search

This repository contains the source code associated with the paper

> CrossBeam: Learning bottom-up search strategies for progam synthesis.

Title and venue subject to change, after the paper is written and published.

This is a research project that aims to develop new methods for learning to
prioritize partial programs within beam search for program synthesis.


## Setup

We use pip package for management. Please first do pip install at the root
folder:

    pip install -e .

You can also append `--user` option if you don't want to install globally.

## Running code

From the root `crossbeam` directory:

```
python3 -m crossbeam.experiment.tuple_synthesis.main
```

## Running tests

From the root `crossbeam` directory:

```
pytest
```

## Disclaimer

This is not an official Google product.
