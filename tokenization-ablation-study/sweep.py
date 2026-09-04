"""
Ablation runner for the mini-GPT: one axis varied at a time
from a shared baseline, everything else held fixed, via train.run_training()
(no duplicated training loop) so every ablation run is directly comparable
to the baseline `python train.py --tag baseline` run.

Axes:
    context   -> block_size in {32, 128, 256}
    depth     -> n_layers  in {2, 4, 6}
    lr        -> lr        in {1e-4, 3e-4, 1e-3}
    seeds     -> baseline config re-run at seed in {42, 123, 2026}, mean/std PPL
    tie       -> tie_weights in {False, True}, params/PPL trade-off
    all       -> all five

Usage:
    python sweep.py --axis context --max_iters 3000
    python sweep.py --axis all --max_iters 3000   # keep small for a first pass
"""
import argparse
import csv
import math
import os
from types import SimpleNamespace

import torch

import train as train_module


BASELINE = dict(
    d_model=192, n_heads=6, n_layers=4, block_size=128, batch_size=64,
    lr=3e-4, max_iters=5000, eval_interval=250, dropout=0.1,
    vocab_size=2000, seed=42, tie_weights=False,
)


def make_args(overrides: dict, out_dir: str, tag: str) -> SimpleNamespace:
    cfg = {**BASELINE, **overrides, "out_dir": out_dir, "tag": tag}
    return SimpleNamespace(**cfg)


def _plot_axis(csv_paths, labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for csv_path, label in zip(csv_paths, labels):
        iters, train_losses, val_losses, val_ppls = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                iters.append(int(row["iter"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                val_ppls.append(float(row["val_perplexity"]))
        axes[0].plot(iters, train_losses, "--", alpha=0.5, label=f"{label} (train)")
        axes[0].plot(iters, val_losses, "-", label=f"{label} (val)")
        axes[1].plot(iters, val_ppls, "-", label=label)

    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("perplexity"); axes[1].set_title("Val Perplexity")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)


def run_axis(axis_name: str, configs: list, out_root: str, max_iters: int):
    axis_dir = os.path.join(out_root, axis_name)
    os.makedirs(axis_dir, exist_ok=True)

    summaries = []
    for tag, overrides in configs:
        run_dir = os.path.join(axis_dir, tag)
        os.makedirs(run_dir, exist_ok=True)
        overrides = {**overrides, "max_iters": max_iters}
        args = make_args(overrides, run_dir, tag)
        print(f"\n--- [{axis_name}] running {tag}: {overrides} ---")
        summaries.append(train_module.run_training(args))

    _plot_axis([s["loss_log"] for s in summaries], [s["tag"] for s in summaries],
               os.path.join(axis_dir, "loss_comparison.png"))

    table_path = os.path.join(axis_dir, "summary.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "n_params", "final_val_loss", "final_val_ppl", "training_time_s"])
        for s in summaries:
            writer.writerow([s["tag"], s["n_params"], s["final_val_loss"], s["final_val_ppl"], s["training_time_s"]])

    print(f"\n[{axis_name}] summary table -> {table_path}")
    print(f"[{axis_name}] comparison plot -> {axis_dir}/loss_comparison.png")
    return summaries


def run_seeds(out_root: str, max_iters: int, seeds=(42, 123, 2026)):
    axis_dir = os.path.join(out_root, "seeds")
    os.makedirs(axis_dir, exist_ok=True)

    summaries = []
    for seed in seeds:
        run_dir = os.path.join(axis_dir, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        args = make_args({"seed": seed, "max_iters": max_iters}, run_dir, f"seed{seed}")
        print(f"\n--- [seeds] running seed={seed} ---")
        summaries.append(train_module.run_training(args))

    ppls = [s["final_val_ppl"] for s in summaries]
    losses = [s["final_val_loss"] for s in summaries]
    mean_ppl, std_ppl = sum(ppls) / len(ppls), torch.tensor(ppls).std().item()
    mean_loss, std_loss = sum(losses) / len(losses), torch.tensor(losses).std().item()

    table_path = os.path.join(axis_dir, "summary.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "final_val_loss", "final_val_ppl"])
        for s in summaries:
            writer.writerow([s["seed"], s["final_val_loss"], s["final_val_ppl"]])
        writer.writerow(["mean", mean_loss, mean_ppl])
        writer.writerow(["std", std_loss, std_ppl])

    print(f"\n[seeds] val PPL: {[round(p, 2) for p in ppls]} -> mean {mean_ppl:.2f} +/- {std_ppl:.2f}")
    print(f"[seeds] summary table -> {table_path}")
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["context", "depth", "lr", "seeds", "tie", "all"], default="all")
    p.add_argument("--max_iters", type=int, default=5000, help="reduce for a fast sanity pass, e.g. 500")
    p.add_argument("--out_root", type=str, default="checkpoints")
    args = p.parse_args()

    if args.axis in ("context", "all"):
        run_axis("context", [
            ("ctx32", {"block_size": 32}),
            ("ctx128", {"block_size": 128}),
            ("ctx256", {"block_size": 256}),
        ], args.out_root, args.max_iters)

    if args.axis in ("depth", "all"):
        run_axis("depth", [
            ("layers2", {"n_layers": 2}),
            ("layers4", {"n_layers": 4}),
            ("layers6", {"n_layers": 6}),
        ], args.out_root, args.max_iters)

    if args.axis in ("lr", "all"):
        run_axis("learning_rate", [
            ("lr1e-4", {"lr": 1e-4}),
            ("lr3e-4", {"lr": 3e-4}),
            ("lr1e-3", {"lr": 1e-3}),
        ], args.out_root, args.max_iters)

    if args.axis in ("tie", "all"):
        run_axis("weight_tying", [
            ("untied", {"tie_weights": False}),
            ("tied", {"tie_weights": True}),
        ], args.out_root, args.max_iters)

    if args.axis in ("seeds", "all"):
        run_seeds(args.out_root, args.max_iters)


if __name__ == "__main__":
    main()
