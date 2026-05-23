"""Run all Project-1 experiments and aggregate results."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "experiment_outputs")
FIG = os.path.join(ROOT, "..", "report", "figures")


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    py = sys.executable

    experiments = [
        [
            py,
            "train_experiment.py",
            "--model",
            "mlp",
            "--scheduler",
            "--epochs",
            "5",
            "--save-dir",
            os.path.join(OUT, "mlp_multistep"),
            "--name",
            "mlp_multistep",
        ],
        [
            py,
            "train_experiment.py",
            "--model",
            "mlp",
            "--no-scheduler",
            "--epochs",
            "5",
            "--save-dir",
            os.path.join(OUT, "mlp_constant_lr"),
            "--name",
            "mlp_constant_lr",
        ],
        [
            py,
            "train_experiment.py",
            "--model",
            "cnn",
            "--scheduler",
            "--epochs",
            "3",
            "--save-dir",
            os.path.join(OUT, "cnn_multistep"),
            "--name",
            "cnn_multistep",
            "--eval-every",
            "50",
            "--dev-max",
            "3000",
        ],
    ]

    summaries = []
    for cmd in experiments:
        run(cmd)
        save_dir = cmd[cmd.index("--save-dir") + 1]
        with open(os.path.join(save_dir, "summary.json"), encoding="utf-8") as f:
            summaries.append(json.load(f))

    import shutil

    for key, fig_name in [
        ("mlp_multistep", "mlp_curves.png"),
        ("cnn_multistep", "cnn_curves.png"),
    ]:
        src = os.path.join(OUT, key, "curves.png")
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(FIG, fig_name))

    mlp_ckpt = os.path.join(OUT, "mlp_multistep", "best_model.pickle")
    cnn_ckpt = os.path.join(OUT, "cnn_multistep", "best_model.pickle")
    plot_cmd = [
        py,
        "plot_analysis.py",
        "--mlp-checkpoint",
        mlp_ckpt,
        "--out-dir",
        FIG,
        "--mlp-constant-history",
        os.path.join(OUT, "mlp_constant_lr", "runner_history.pickle"),
        "--mlp-scheduler-history",
        os.path.join(OUT, "mlp_multistep", "runner_history.pickle"),
    ]
    if os.path.isfile(cnn_ckpt):
        plot_cmd.extend(["--cnn-checkpoint", cnn_ckpt])
    run(plot_cmd)

    results = {s["name"]: s for s in summaries}
    results_path = os.path.join(OUT, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll done. Results: {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
