import os
import urllib.request

import torch

from tokenizer import train_tokenizer, load_tokenizer, TOKENIZER_DIR

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "input.txt")


def download_corpus(path: str = CORPUS_PATH, url: str = CORPUS_URL) -> str:
    """Fetch the corpus if it is not already on disk.

    This project originally assumed `input.txt` was already present, because it
    was during development -- another checkout had downloaded it into a sibling
    directory. On a fresh clone (where input.txt is gitignored, as downloaded
    data should be) the tokenizer trainer failed with a bare
    "No such file or directory" from inside the Rust tokenizers library, which
    gives no hint about the actual cause.
    """
    if not os.path.exists(path):
        print(f"downloading corpus from {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def get_tokenizer(vocab_size: int = 2000):
    download_corpus()
    # a previously-cached tokenizer is only reused if it actually matches the
    # requested vocab_size -- otherwise load_tokenizer() would silently hand
    # back a stale tokenizer trained at a different size (a real bug caught
    # while testing sweep.py: it takes changing --vocab_size between runs to
    # trigger, which the original single-run train.py usage never exercised).
    try:
        tok = load_tokenizer()
        if tok.get_vocab_size() != vocab_size:
            return train_tokenizer(CORPUS_PATH, vocab_size=vocab_size)
        return tok
    except FileNotFoundError:
        return train_tokenizer(CORPUS_PATH, vocab_size=vocab_size)


def prepare_dataset(split: float = 0.9, vocab_size: int = 2000):
    download_corpus()
    tok = get_tokenizer(vocab_size)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)
    n = int(split * len(ids))
    return ids[:n], ids[n:], tok


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    train_data, val_data, tok = prepare_dataset()
    print(f"train tokens: {len(train_data)} | val tokens: {len(val_data)} | vocab: {tok.get_vocab_size()}")
