"""
Figures generated from the committed results.

    fig_reversal.png   the headline: two ablation conclusions that invert
                       between 500 and 5000 iterations, shown as curves so the
                       crossing point is visible rather than asserted.

    fig_ablations.png  all four axes as final-perplexity bars against the
                       measured seed sigma.

    python make_figures.py
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "checkpoints"


def curve(axis, tag):
    p = f"{ROOT}/{axis}/{tag}/loss_log_{tag}.csv"
    rows = list(csv.DictReader(open(p)))
    return ([int(r["iter"]) for r in rows],
            [float(r["val_perplexity"]) for r in rows])


def seed_sigma():
    rows = list(csv.DictReader(open(f"{ROOT}/seeds/summary.csv")))
    return float(next(r["final_val_ppl"] for r in rows if r["seed"] == "std"))


def fig_reversal():
    sigma = seed_sigma()
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- learning rate: the clean reversal -------------------------------
    for tag, col in (("lr1e-4", "#999"), ("lr3e-4", "#4c72b0"), ("lr1e-3", "#c44e52")):
        it, p = curve("learning_rate", tag)
        axs[0].plot(it[1:], p[1:], "-", color=col, label=tag)
    axs[0].axvline(500, ls=":", color="black", lw=1.2)
    axs[0].annotate("ablation stopped here\nin the first run",
                    (560, 260), fontsize=8)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("val perplexity (log)")
    axs[0].set_title("Learning rate: 1e-3 wins at 500, loses badly at 5000")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3, which="both")

    # --- weight tying ratio ---------------------------------------------
    itu, pu = curve("weight_tying", "untied")
    itt, pt = curve("weight_tying", "tied")
    ratio = [t / u for t, u in zip(pt, pu)]
    axs[1].plot(itu, ratio, "o-", color="#c44e52", ms=4)
    axs[1].axhline(1.0, ls="--", color="grey", lw=1)
    axs[1].axvline(500, ls=":", color="black", lw=1.2)
    peak = max(range(len(ratio)), key=lambda i: ratio[i])
    axs[1].annotate(f"peak {ratio[peak]:.2f}x", (itu[peak], ratio[peak]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    axs[1].annotate(f"{ratio[-1]:.2f}x", (itu[-1], ratio[-1]),
                    textcoords="offset points", xytext=(-30, -14), fontsize=9)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("tied PPL / untied PPL")
    axs[1].set_title("Weight tying: penalty peaks mid-training, then shrinks")
    axs[1].grid(alpha=0.3)

    fig.suptitle("Two ablation conclusions that depend on when you stop "
                 f"(seed sigma = {sigma:.2f} PPL)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/fig_reversal.png", dpi=150)
    print(f"saved {ROOT}/fig_reversal.png")


def fig_ablations():
    sigma = seed_sigma()
    axes_ = ["context", "depth", "learning_rate", "weight_tying"]
    titles = ["Context length", "Depth", "Learning rate", "Weight tying"]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, axis, title in zip(axs, axes_, titles):
        rows = [r for r in csv.DictReader(open(f"{ROOT}/{axis}/summary.csv"))]
        tags = [r["tag"] for r in rows]
        ppls = [float(r["final_val_ppl"]) for r in rows]
        best = min(ppls)
        cols = ["#4c72b0" if abs(p - best) < 3 * sigma else "#c44e52" for p in ppls]
        ax.bar(tags, ppls, color=cols)
        ax.axhspan(best - sigma, best + sigma, color="grey", alpha=0.25, zorder=0)
        ax.set_title(title, fontsize=11)
        if axis == "context":
            ax.set_ylabel("val perplexity")
        ax.set_ylim(min(ppls) - 6, max(ppls) + 8)
        ax.grid(alpha=0.3, axis="y")
        for i, p in enumerate(ppls):
            ax.text(i, p + 1.2, f"{p:.1f}", ha="center", fontsize=9)

    fig.suptitle(f"Ablations at 5000 iters — grey band = ±1σ seed noise "
                 f"(σ = {sigma:.2f} PPL); red = >3σ from best", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/fig_ablations.png", dpi=150)
    print(f"saved {ROOT}/fig_ablations.png")


if __name__ == "__main__":
    fig_reversal()
    fig_ablations()
