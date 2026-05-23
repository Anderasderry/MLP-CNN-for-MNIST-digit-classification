# MLP-CNN-for-MNIST-digit-classification

Project 1 for Neural Networks and Deep Learning (SDS, Fudan, Spring 2026).

NumPy-only MLP and CNN for MNIST digit classification. MNIST raw data and model checkpoints are **not** in this repo (see `.gitignore`). Run `data_utils.py` or any training script to download MNIST into `dataset/MNIST/`. Upload trained weights to ModelScope as required by the course.

## Quick start

1. Explore data: `dataset_explore.ipynb`
2. Train: `python test_train.py` or `python run_all_experiments.py`
3. Evaluate: `python test_model.py` (set checkpoint path in the script)

## Project layout

| Path | Description |
|------|-------------|
| `mynn/op.py` | `Linear`, `MultiCrossEntropyLoss`, `conv2D` |
| `mynn/models.py` | `Model_MLP`, `Model_CNN` |
| `mynn/lr_scheduler.py` | e.g. `MultiStepLR` |
| `train_experiment.py` | Single experiment run |
| `run_all_experiments.py` | MLP/CNN + comparison plots |

## Original starter notes

### Codes implemented

1. `op.py` — `Linear`, `MultiCrossEntropyLoss`, `conv2D`
2. `models.py` — MLP / CNN structure
3. `mynn/lr_scheduler.py` — learning rate schedulers
4. `optimizer.py` — SGD (in-place updates)

### Train the model

Open `test_train.py`, modify parameters, and run.

### Test the model

Open `test_model.py`, specify the saved checkpoint path, then run.
