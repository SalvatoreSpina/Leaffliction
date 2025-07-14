# Leaffliction

Leaffliction provides a set of tools for analysing plant leaf images and training
models to detect diseases. It bundles utilities for exploring the dataset,
applying augmentations, performing image transformations and training a simple
CNN classifier.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Alternatively run `bash setup.sh` to install the dependencies in a local folder
and configure the `PYTHONPATH` as explained in the script output.

## Project structure

```
leaffliction/
    analysis/           # dataset statistics utilities
    augmentation/       # image augmentation helpers
    transforms/         # image transformation logic
    model/              # training and prediction scripts
```

Each subpackage exposes scripts that can be run directly with `python3`.

## Usage examples

### Analyse dataset distribution

```bash
python3 leaffliction/analysis/distribution.py path/to/images --save
```

### Create balanced dataset

```bash
python3 leaffliction/augmentation/augmentation.py path/to/images
```

### Apply transformations to images

```bash
python3 leaffliction/transforms/transformation.py -src images -dst out -all
```

### Train and evaluate model

```bash
python3 leaffliction/model/train.py path/to/training
python3 leaffliction/model/predict.py path/to/training path/to/image.jpg
```

The `model/full_dataset.sh` script demonstrates an end‑to‑end workflow that
splits a dataset, augments images and trains a network.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
more information.
