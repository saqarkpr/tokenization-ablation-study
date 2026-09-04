"""
Turns loss_log_*.csv files (written by train.py, which already include a
val_perplexity column) into PNG plots. Mirrors
a character-level equivalent.

Usage:
    python plot_results.py --csv checkpoints/loss_log_baseline.csv --out results/baseline/loss.png
    python plot_results.py --csv checkpoints/context/ctx32/loss_log_ctx32.csv \
                                  checkpoints/context/ctx128/loss_log_ctx128.csv \
                            --labels "block_size=32" "block_size=128" \
                            --out checkpoints/context/loss_comparison.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    iters, train_losses, val_losses, val_ppls = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_ppls.append(float(row["val_perplexity"]))
    return iters, train_losses, val_losses, val_ppls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, nargs="+", required=True)
    p.add_argument("--labels", type=str, nargs="+", default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--metric", choices=["loss", "perplexity", "both"], default="both")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(c) for c in args.csv]
    assert len(labels) == len(args.csv), "--labels must match --csv in count"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_panels = 2 if args.metric == "both" else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for csv_path, label in zip(args.csv, labels):
        iters, train_losses, val_losses, val_ppls = read_log(csv_path)
        panel = 0
        if args.metric in ("loss", "both"):
            axes[panel].plot(iters, train_losses, "--", alpha=0.5, label=f"{label} (train)")
            axes[panel].plot(iters, val_losses, "-", label=f"{label} (val)")
            panel += 1
        if args.metric in ("perplexity", "both"):
            axes[panel].plot(iters, val_ppls, "-", label=label)

    panel = 0
    if args.metric in ("loss", "both"):
        axes[panel].set_xlabel("iteration"); axes[panel].set_ylabel("cross-entropy loss")
        axes[panel].set_title("Training / Validation Loss"); axes[panel].legend(fontsize=8); axes[panel].grid(alpha=0.3)
        panel += 1
    if args.metric in ("perplexity", "both"):
        axes[panel].set_xlabel("iteration"); axes[panel].set_ylabel("perplexity")
        axes[panel].set_title("Validation Perplexity"); axes[panel].legend(fontsize=8); axes[panel].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
