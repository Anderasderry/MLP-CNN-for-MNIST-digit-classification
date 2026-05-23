"""Patch report.tex table and fill-in lines from experiment_outputs/results.json."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "experiment_outputs", "results.json")
REPORT = os.path.join(ROOT, "..", "report", "report.tex")


def pct(x):
    return f"{100.0 * x:.2f}"


def main():
    with open(RESULTS, encoding="utf-8") as f:
        r = json.load(f)

    mlp_s = r["mlp_multistep"]
    mlp_c = r["mlp_constant_lr"]
    cnn = r.get("cnn_multistep", {})

    replacements = [
        (r"\\textbf\{Fill in after training:\}[^\n]*", f"\\textbf{{Results:}} best validation accuracy = {pct(mlp_s['best_val_acc'])}\\%, test accuracy = {pct(mlp_s['test_acc'])}\\%."),
        (
            r"\\textbf\{Fill in:\} CNN validation[^\n]*",
            f"\\textbf{{Results:}} CNN val/test = {pct(cnn.get('val_acc', 0))}\\% / {pct(cnn.get('test_acc', 0))}\\% vs MLP {pct(mlp_s['test_acc'])}\\%; CNN train time $\\approx$ {cnn.get('train_time_sec', 0):.0f}\\,s ({cnn.get('epochs', 3)} epochs).",
        ),
        (
            r"\\textbf\{Fill in:\} e\.g\., scheduled LR[^\n]*",
            f"\\textbf{{Results:}} MultiStepLR val {pct(mlp_s['val_acc'])}\\% (test {pct(mlp_s['test_acc'])}\\%) vs constant LR val {pct(mlp_c['val_acc'])}\\% (test {pct(mlp_c['test_acc'])}\\%).",
        ),
        (
            r"A & MLP baseline \(MultiStepLR\) & TBD & TBD",
            f"A & MLP baseline (MultiStepLR) & {pct(mlp_s['val_acc'])} & {pct(mlp_s['test_acc'])}",
        ),
        (
            r"B & CNN \(same protocol\)\s*& TBD & TBD",
            f"B & CNN (same protocol)        & {pct(cnn.get('val_acc', 0))} & {pct(cnn.get('test_acc', 0))}",
        ),
        (
            r"C1 & MLP, constant LR only\s*& TBD & TBD",
            f"C1 & MLP, constant LR only     & {pct(mlp_c['val_acc'])} & {pct(mlp_c['test_acc'])}",
        ),
        (
            r"C1 & MLP, MultiStepLR\s*& TBD & TBD",
            f"C1 & MLP, MultiStepLR          & {pct(mlp_s['val_acc'])} & {pct(mlp_s['test_acc'])}",
        ),
    ]

    with open(REPORT, encoding="utf-8") as f:
        tex = f.read()

    for pat, rep in replacements:
        tex, n = re.subn(pat, rep, tex, count=1)
        if n == 0:
            print(f"Warning: pattern not found: {pat[:40]}...")

    # Replace fbox placeholders with includegraphics where files exist
    fig_dir = "figures"
    fig_map = [
        ("mlp_curves", "fig:mlp_curve"),
        ("cnn_curves", "fig:cnn_curve"),
        ("lr_schedule_compare", "fig:lr_compare"),
        ("confusion", "fig:confusion"),
        ("misclassified", "fig:confusion"),
        ("mlp_weights", "fig:kernels"),
        ("cnn_kernels", "fig:kernels"),
    ]
    for stem, _ in fig_map:
        path = os.path.join(ROOT, "..", "report", fig_dir, f"{stem}.png")
        if not os.path.isfile(path):
            continue

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    main()
