# MLP-CNN-for-MNIST-digit-classification

**Project 1** for *Neural Networks and Deep Learning* (SDS, Fudan University, Spring 2026).

A **NumPy-only** MLP and CNN for MNIST digit classification.

- All runnable code is under [`codes/`](codes/).
- This repo does **not** include MNIST raw data or model weights (see [`codes/.gitignore`](codes/.gitignore)).
- **Trained weights** are on [ModelScope](https://www.modelscope.cn/models/Anderasderry/MLP_CNN_for_MNIST_classification) (see [§ Model weights](#model-weights-modelscope) below).

**Requirements:** Python 3.8+, `numpy`, `matplotlib`

```bash
pip install numpy matplotlib
```

**Clone and enter the code directory:**

```bash
git clone https://github.com/Anderasderry/MLP-CNN-for-MNIST-digit-classification.git
cd MLP-CNN-for-MNIST-digit-classification/codes
```

All commands below are run from the **`codes/`** directory.

---

## 1. How to download the MNIST dataset

Data are stored in `codes/dataset/MNIST/` as four `.gz` files:

| File | Content |
|------|---------|
| `train-images-idx3-ubyte.gz` | 60,000 training images |
| `train-labels-idx1-ubyte.gz` | training labels |
| `t10k-images-idx3-ubyte.gz` | 10,000 test images |
| `t10k-labels-idx1-ubyte.gz` | test labels |

Start training directly. `data_utils.load_mnist_splits()` downloads any missing files from the public MNIST mirror:

```bash
cd codes
python run_all_experiments.py
```

Download only (no training):

```bash
cd codes
python -c "from data_utils import download_mnist; download_mnist()"
```

### Data split

- Official 60k training set → **50k train + 10k validation** (fixed seed `309`; split cached in `codes/idx.pickle`)
- Official 10k test set → used only for **final test accuracy** after training

---

## 2. How to run training

### Run all experiments (recommended, for report reproduction)

```bash
cd codes
python run_all_experiments.py
```

This runs **three** training jobs in sequence, writes `experiment_outputs/results.json`, then calls `plot_analysis.py` and copies some figures to `report/figures/`.

### Run a single experiment

```bash
cd codes

# Experiment 1: MLP + MultiStepLR
python train_experiment.py --model mlp --scheduler --epochs 5 \
  --save-dir ./experiment_outputs/mlp_multistep --name mlp_multistep

# Experiment 2: MLP + fixed learning rate
python train_experiment.py --model mlp --no-scheduler --epochs 5 \
  --save-dir ./experiment_outputs/mlp_constant_lr --name mlp_constant_lr

# Experiment 3: CNN + MultiStepLR
python train_experiment.py --model cnn --scheduler --epochs 3 \
  --save-dir ./experiment_outputs/cnn_multistep --name cnn_multistep \
  --eval-every 50 --dev-max 3000
```

Shortcut (Experiment 1 only):

```bash
python test_train.py
```

### Evaluate a saved checkpoint

```bash
python test_model.py --model mlp \
  --checkpoint ./experiment_outputs/mlp_multistep/best_model.pickle

python test_model.py --model cnn \
  --checkpoint ./experiment_outputs/cnn_multistep/best_model.pickle
```

---

## 3. What the training runs are

`run_all_experiments.py` runs these **three** experiments (Part B models + Part C optimization comparison):

| Run directory | Model | Learning rate | Epochs | Role in the project |
|---------------|-------|---------------|--------|---------------------|
| `mlp_multistep` | MLP: `784 → 600 → 10`, ReLU | **MultiStepLR** (×0.5 at steps 800 / 2400 / 4000) | 5 | Main MLP baseline; MLP curves and weight visualization in the report |
| `mlp_constant_lr` | Same MLP | **Fixed lr = 0.06** (no scheduler) | 5 | Compare MultiStepLR vs constant learning rate |
| `cnn_multistep` | CNN: 2 conv layers + ReLU + linear | **MultiStepLR** | 3 | CNN baseline; dev eval every 50 steps on up to 3000 validation images for speed |

**Shared hyperparameters:**

- Optimizer: SGD, initial lr `0.06`, batch size `32`
- Weight decay: `1e-4`
- Loss: `MultiCrossEntropyLoss` (softmax + cross-entropy)
- Framework: pure NumPy on CPU (**no GPU**)

**Outputs per run** (`codes/experiment_outputs/<run_name>/`):

| File | Description |
|------|-------------|
| `best_model.pickle` | Best checkpoint on validation (**upload to ModelScope; do not push to GitHub**) |
| `summary.json` | Train / dev / test metrics |
| `curves.png` | Training and validation loss & accuracy |
| `runner_history.pickle` | Full training history (e.g. LR comparison plots) |

Aggregated metrics: `codes/experiment_outputs/results.json`.

---

## Model weights (ModelScope)

Pre-trained checkpoints (NumPy `pickle`, produced by `Model_MLP.save_model` / `Model_CNN.save_model`):

**ModelScope:** https://www.modelscope.cn/models/Anderasderry/MLP_CNN_for_MNIST_classification

After downloading from ModelScope, place the files under `codes/experiment_outputs/` using the same layout as local training:

```
codes/experiment_outputs/
├── mlp_multistep/
│   └── best_model.pickle      # MLP + MultiStepLR (val 94.96%, test 95.41%)
├── mlp_constant_lr/
│   └── best_model.pickle      # MLP + fixed LR (val 96.86%, test 97.31%)
└── cnn_multistep/
    └── best_model.pickle      # CNN + MultiStepLR (val 98.44%, test 98.53%)
```

| File | Model | Use with |
|------|-------|----------|
| `mlp_multistep/best_model.pickle` | MLP baseline (MultiStepLR) | `test_model.py --model mlp`, `plot_analysis.py` (default) |
| `mlp_constant_lr/best_model.pickle` | MLP, fixed learning rate | Part C optimization comparison |
| `cnn_multistep/best_model.pickle` | CNN baseline | `test_model.py --model cnn`, CNN visualizations |

**Load and evaluate** (from `codes/`):

```bash
python test_model.py --model mlp --checkpoint ./experiment_outputs/mlp_multistep/best_model.pickle
python test_model.py --model cnn --checkpoint ./experiment_outputs/cnn_multistep/best_model.pickle
```

**Regenerate report figures** (requires all three checkpoints above):

```bash
python plot_analysis.py \
  --mlp-checkpoint ./experiment_outputs/mlp_multistep/best_model.pickle \
  --cnn-checkpoint ./experiment_outputs/cnn_multistep/best_model.pickle \
  --mlp-constant-history ./experiment_outputs/mlp_constant_lr/runner_history.pickle \
  --mlp-scheduler-history ./experiment_outputs/mlp_multistep/runner_history.pickle \
  --out-dir ../report/figures
```

> `runner_history.pickle` files are optional on ModelScope; they are only needed for the learning-rate comparison plot. Re-run `run_all_experiments.py` locally if you need them.

---

## Code layout (`codes/`)

| Path | Description |
|------|-------------|
| `mynn/op.py` | `Linear`, `MultiCrossEntropyLoss`, `conv2D` |
| `mynn/models.py` | `Model_MLP`, `Model_CNN` |
| `mynn/optimizer.py` / `lr_scheduler.py` / `runner.py` | SGD, `MultiStepLR`, training loop |
| `data_utils.py` | MNIST download and splits |
| `train_experiment.py` | Single-experiment entry point |
| `run_all_experiments.py` | All three experiments + aggregation |
| `plot_analysis.py` | Confusion matrix, misclassified samples, weight/kernel plots |

Optional: explore data in `dataset_explore.ipynb`.

---

## Notes

- **Runtime:** MLP ~tens of minutes per 5 epochs on CPU; CNN is slower. Full `run_all_experiments.py` may take about **1–2 hours** depending on hardware.
- **Submission:** code on GitHub; weights on [ModelScope](https://www.modelscope.cn/models/Anderasderry/MLP_CNN_for_MNIST_classification). `*.pickle` files are listed in `.gitignore`.
