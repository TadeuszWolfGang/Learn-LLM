"""
Dane i tokenizer znakowy.  Wspólne dla wszystkich lekcji.

Tokenizer char-level: każdy unikalny znak w pliku = jeden token.
Tiny Shakespeare ma 65 znaków -> vocab_size = 65.
Prawdziwe LLM-y używają BPE (podsłowa, vocab ~50k-150k), ale mechanika
modelu jest identyczna, zmienia się tylko vocab_size.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "..", "data", "input.txt")


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def load_text(path=DEFAULT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Brak {path}. Uruchom najpierw: python3 00_check_env.py (pobiera dane)")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_dataset(path=DEFAULT_PATH, val_frac=0.1):
    """Zwraca (tokenizer, train_ids, val_ids).  Ostatnie 10% tekstu = walidacja."""
    text = load_text(path)
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    n = int(len(ids) * (1 - val_frac))
    return tok, ids[:n], ids[n:]


def get_batch(ids, block_size, batch_size, rng=np.random):
    """Losowe okna długości block_size.  y to x przesunięte o 1 znak w prawo:
       x = "Before we", y = "efore we " -> model uczy się przewidywać NASTĘPNY znak."""
    ix = rng.randint(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[i:i + block_size] for i in ix])
    y = np.stack([ids[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def download_if_missing(path=DEFAULT_PATH):
    if os.path.exists(path):
        return path
    import urllib.request
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Pobieram Tiny Shakespeare (~1 MB)...")
    urllib.request.urlretrieve(url, path)
    return path
