# Train MLP (default). For full pipeline use: python run_all_experiments.py
import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call(
        [
            sys.executable,
            "train_experiment.py",
            "--model",
            "mlp",
            "--scheduler",
            "--epochs",
            "5",
            "--save-dir",
            "./experiment_outputs/mlp_multistep",
            "--name",
            "mlp_multistep",
        ],
        cwd=sys.path[0] or ".",
    )
