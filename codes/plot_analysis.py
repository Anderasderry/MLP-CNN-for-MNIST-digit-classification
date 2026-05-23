"""Generate Part-C visualization figures for the report."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from data_utils import load_mnist_splits


def plot_confusion_and_misclassified(model, test_imgs, test_labs, out_dir, model_type, n_show=16):
    logits = model(test_imgs)
    preds = np.argmax(logits, axis=1)
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(test_labs, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix (counts)")
    fig.savefig(os.path.join(out_dir, "confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    wrong_idx = np.where(preds != test_labs)[0][:n_show]
    ncol = 4
    nrow = int(np.ceil(len(wrong_idx) / ncol)) or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = np.array(axes).reshape(-1)
    for k, idx in enumerate(wrong_idx):
        img = test_imgs[idx].reshape(28, 28)
        axes[k].imshow(img, cmap="gray")
        axes[k].set_title(f"t={test_labs[idx]}, p={preds[idx]}")
        axes[k].axis("off")
    for k in range(len(wrong_idx), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Misclassified examples")
    fig.savefig(os.path.join(out_dir, "misclassified.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_mlp_weights(model, out_dir, n_show=64):
    W = model.layers[0].params["W"]  # [784, 600]
    ncol = 8
    nrow = n_show // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.2, nrow * 1.2))
    axes = axes.reshape(-1)
    for i in range(n_show):
        axes[i].imshow(W[:, i].reshape(28, 28), cmap="viridis")
        axes[i].axis("off")
    fig.suptitle("MLP first-layer weight patches (subset)")
    fig.savefig(os.path.join(out_dir, "mlp_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cnn_kernels(model, out_dir):
    conv = model.layers[0]
    W = conv.W  # [out, in, kh, kw]
    n = W.shape[0]
    ncol = 8
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.2, nrow * 1.2))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        k = W[i, 0]
        axes[i].imshow(k, cmap="viridis")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle("CNN conv1 filters")
    fig.savefig(os.path.join(out_dir, "cnn_kernels.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr_compare(curve_paths, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path, label in zip(curve_paths, labels):
        import pickle

        with open(path, "rb") as f:
            hist = pickle.load(f)
        ax.plot(hist["dev_scores"], label=label)
    ax.set_xlabel("eval step")
    ax.set_ylabel("dev accuracy")
    ax.legend()
    ax.set_title("Optimization: dev accuracy comparison")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--out-dir", default="../report/figures")
    parser.add_argument("--mlp-constant-history", default=None)
    parser.add_argument("--mlp-scheduler-history", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _, _, test_set = load_mnist_splits()
    test_imgs, test_labs = test_set

    mlp = nn.models.Model_MLP()
    mlp.load_model(args.mlp_checkpoint)
    plot_confusion_and_misclassified(mlp, test_imgs, test_labs, args.out_dir, "mlp")
    plot_mlp_weights(mlp, args.out_dir)

    if args.cnn_checkpoint:
        cnn = nn.models.Model_CNN()
        cnn.load_model(args.cnn_checkpoint)
        plot_cnn_kernels(cnn, args.out_dir)

    if args.mlp_constant_history and args.mlp_scheduler_history:
        plot_lr_compare(
            [args.mlp_constant_history, args.mlp_scheduler_history],
            ["Constant LR", "MultiStepLR"],
            os.path.join(args.out_dir, "lr_schedule_compare.png"),
        )

    print(f"Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
