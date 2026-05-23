"""Evaluate saved checkpoint on MNIST test set."""
import argparse
import sys

import mynn as nn
from data_utils import load_mnist_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument(
        "--checkpoint",
        default=r".\experiment_outputs\mlp_multistep\best_model.pickle",
    )
    args = parser.parse_args()

    if args.model == "mlp":
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model.load_model(args.checkpoint)

    _, _, test_set = load_mnist_splits()
    logits = model(test_set[0])
    print(nn.metric.accuracy(logits, test_set[1]))


if __name__ == "__main__":
    main()
