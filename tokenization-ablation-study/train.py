"""
Training loop for the BPE-tokenized mini-GPT. Same underlying architecture
as a character-level decoder-only Transformer, but trained with a real subword
vocabulary and reporting perplexity (exp(val_loss)) in addition to loss,
since perplexity is the standard metric reported for GPT-style LMs.

Structure: `run_training(args)`
is factored out of `main()` so sweep.py can call it in-process for every grid
point instead of duplicating the training loop.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

from data import prepare_dataset, get_batch
from model import TransformerFromScratch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--vocab_size", type=int, default=2000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--tag", type=str, default="run1")
    p.add_argument("--seed", type=int, default=42, help="random seed, for reproducibility / seed-variance study")
    p.add_argument("--tie_weights", action="store_true", help="tie lm_head weights to the token embedding")
    return p


def run_training(args) -> dict:
    """Runs one full training job from a populated args namespace and
    returns a summary dict. Factored out of main() so sweep.py can call
    this directly (in-process) for every grid point."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_data, val_data, tok = prepare_dataset(vocab_size=args.vocab_size)

    model = TransformerFromScratch(
        vocab_size=tok.get_vocab_size(),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.block_size,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(device)

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[{args.tag}] model params: {n_params/1e6:.2f}M | vocab: {tok.get_vocab_size()} | device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = os.path.join(args.out_dir, f"loss_log_{args.tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss", "val_perplexity", "elapsed_s"])

    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0 or it == 1:
            val_loss, val_ppl = evaluate(model, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            print(f"[{args.tag}] iter {it:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | "
                  f"val_ppl {val_ppl:.2f} | {elapsed:.1f}s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, loss.item(), val_loss, val_ppl, elapsed])

    ckpt_path = os.path.join(args.out_dir, f"model_{args.tag}.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)

    final_val_loss, final_val_ppl = evaluate(model, val_data, args.block_size, args.batch_size, device, eval_iters=50)
    total_elapsed = time.time() - t0

    summary = {
        "tag": args.tag,
        "seed": args.seed,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "block_size": args.block_size,
        "lr": args.lr,
        "tie_weights": args.tie_weights,
        "vocab_size": args.vocab_size,
        "max_iters": args.max_iters,
        "n_params": n_params,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "training_time_s": total_elapsed,
        "checkpoint": ckpt_path,
        "loss_log": log_path,
    }
    summary_path = os.path.join(args.out_dir, f"summary_{args.tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{args.tag}] saved checkpoint to {ckpt_path}, summary to {summary_path}")
    return summary


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
