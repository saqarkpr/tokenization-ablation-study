"""
Sample text from a trained mini-GPT checkpoint using the BPE tokenizer.
Supports multiple temperatures in one call, same seed across each, for a
systematic qualitative comparison.

    python generate.py --ckpt checkpoints/model_baseline.pt --prompt "ROMEO:" \
        --temperature 0.7 1.0 1.3 --max_new_tokens 200 --seed 42
"""
import argparse
import torch

from model import TransformerFromScratch
from data import get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, nargs="+", default=[0.8],
                    help="one or more temperatures; generates one sample per value")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None, help="fix the sampling seed for reproducible comparisons")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    # load the SAME vocab_size tokenizer the checkpoint was trained with --
    # not just whatever tokenizer happens to be cached on disk (see the fix
    # in data.py's get_tokenizer(); generate.py had the same latent bug).
    tok = get_tokenizer(vocab_size=margs.get("vocab_size", 2000))

    model = TransformerFromScratch(
        vocab_size=tok.get_vocab_size(),
        d_model=margs["d_model"], n_heads=margs["n_heads"], n_layers=margs["n_layers"],
        max_len=margs["block_size"], dropout=0.0, tie_weights=margs.get("tie_weights", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long, device=device)

    for temp in args.temperature:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        out = model.generate(idx, args.max_new_tokens, temperature=temp, top_k=args.top_k)
        text = tok.decode(out[0].tolist())
        if len(args.temperature) > 1:
            print(f"\n=== temperature = {temp} ===")
        print(text)


if __name__ == "__main__":
    main()
