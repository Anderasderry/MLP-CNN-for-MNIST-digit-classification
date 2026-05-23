"""Evaluate a saved MLP or CNN checkpoint on the MNIST test set."""
import argparse
import json
import os

import mynn as nn
from data_utils import load_mnist_splits


def load_checkpoint(path, model_type):
    if model_type == "mlp":
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model.load_model(path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    _, _, test_set = load_mnist_splits()
    model = load_checkpoint(args.checkpoint, args.model)
    logits = model(test_set[0])
    acc = nn.metric.accuracy(logits, test_set[1])
    print(f"Test accuracy: {acc:.5f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"test_acc": float(acc), "checkpoint": args.checkpoint}, f, indent=2)


if __name__ == "__main__":
    main()
