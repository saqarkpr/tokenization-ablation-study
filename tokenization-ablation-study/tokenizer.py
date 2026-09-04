"""
Trains a small byte-pair-encoding (BPE) tokenizer on the local corpus using
HuggingFace's `tokenizers` library (the training algorithm only — this
project does not use `transformers` models/weights). BPE gives a much
smaller, more realistic token vocabulary than a character-level one, which
is the point: most language models operate on subword tokens, and this
project's experiments are meant to reflect that.
"""
import os
from tokenizers import ByteLevelBPETokenizer

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "tokenizer_files")


def train_tokenizer(corpus_path: str, vocab_size: int = 2000, out_dir: str = TOKENIZER_DIR):
    os.makedirs(out_dir, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(files=[corpus_path], vocab_size=vocab_size, min_frequency=2,
              special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok.save_model(out_dir)
    return tok


def load_tokenizer(out_dir: str = TOKENIZER_DIR) -> ByteLevelBPETokenizer:
    vocab_path = os.path.join(out_dir, "vocab.json")
    merges_path = os.path.join(out_dir, "merges.txt")
    if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        raise FileNotFoundError(
            f"No trained tokenizer in {out_dir}. Run train_tokenizer() first."
        )
    return ByteLevelBPETokenizer(vocab_path, merges_path)


if __name__ == "__main__":
    from data import download_corpus
    corpus_path = download_corpus()
    tok = train_tokenizer(corpus_path, vocab_size=2000)
    ids = tok.encode("To be, or not to be, that is the question:").ids
    print("vocab size:", tok.get_vocab_size())
    print("encoded:", ids)
    print("decoded:", tok.decode(ids))
