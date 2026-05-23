"""Train MLP or CNN with configurable optimizer / scheduler."""
import argparse
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from data_utils import load_mnist_splits
from draw_tools.plot import plot


def build_model(model_type, num_features=784, num_classes=10):
    if model_type == "mlp":
        return nn.models.Model_MLP([num_features, 600, num_classes], "ReLU", [1e-4, 1e-4])
    if model_type == "cnn":
        return nn.models.Model_CNN(in_channels=1, num_classes=num_classes, lambda_list=[1e-4, 1e-4, 1e-4])
    raise ValueError(f"Unknown model_type: {model_type}")


def build_runner(model, use_scheduler, init_lr=0.06):
    optimizer = nn.optimizer.SGD(init_lr=init_lr, model=model)
    scheduler = None
    if use_scheduler:
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5
        )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    return nn.runner.RunnerM(
        model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, batch_size=32
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--scheduler", action="store_true", help="Use MultiStepLR")
    parser.add_argument("--no-scheduler", dest="scheduler", action="store_false")
    parser.set_defaults(scheduler=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--log-iters", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="./experiment_outputs/mlp_default")
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--dev-max", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(309)
    train_set, valid_set, test_set = load_mnist_splits()
    train_imgs, train_labs = train_set
    num_classes = int(train_labs.max()) + 1

    model = build_model(args.model, train_imgs.shape[-1], num_classes)
    runner = build_runner(model, args.scheduler, args.lr)

    if args.model == "cnn":
        dev_max = args.dev_max if args.dev_max else 3000
        eval_every = max(args.eval_every, 50)
    else:
        dev_max = args.dev_max
        eval_every = args.eval_every

    t0 = time.time()
    runner.train(
        train_set,
        valid_set,
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        save_dir=args.save_dir,
        dev_max_samples=dev_max,
        eval_every=eval_every,
    )
    train_time = time.time() - t0

    model_path = os.path.join(args.save_dir, "best_model.pickle")
    test_score, test_loss = runner.evaluate(test_set)
    valid_score, valid_loss = runner.evaluate(valid_set)

    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, "runner_history.pickle")
    with open(history_path, "wb") as f:
        pickle.dump(
            {
                "train_loss": runner.train_loss,
                "dev_loss": runner.dev_loss,
                "train_scores": runner.train_scores,
                "dev_scores": runner.dev_scores,
            },
            f,
        )

    curve_path = os.path.join(args.save_dir, "curves.png")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot(runner, axes)
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: could not save learning curves ({e}). History still in runner_history.pickle.")

    summary = {
        "name": args.name,
        "model": args.model,
        "scheduler": args.scheduler,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_acc": float(runner.best_score),
        "val_acc": float(valid_score),
        "val_loss": float(valid_loss),
        "test_acc": float(test_score),
        "test_loss": float(test_loss),
        "train_time_sec": train_time,
        "model_path": model_path,
        "curves_path": curve_path,
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
