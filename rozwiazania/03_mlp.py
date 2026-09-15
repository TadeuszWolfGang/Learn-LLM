"""
LEKCJA 3 - MLP language model (Bengio 2003): kontekst = ostatnie K znaków.

  idx (B, K) -> embedding (B, K, C) -> spłaszcz (B, K*C) -> tanh(W1) -> W2 -> logity (B, V)

Pierwszy raz pojawiają się: embedding, warstwa ukryta, i realny overfitting.

    python3 03_mlp.py                 # K=8, ~3 min
    python3 03_mlp.py --K 3           # mniejszy kontekst -> gorszy loss
    python3 03_mlp.py --train_chars 5000   # mało danych -> overfitting na własne oczy
"""
import argparse
import numpy as np
from autograd import Tensor, cross_entropy, AdamW
from data import download_if_missing, load_dataset

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=8, help="ile znaków kontekstu")
p.add_argument("--C", type=int, default=16, help="wymiar embeddingu")
p.add_argument("--H", type=int, default=256, help="warstwa ukryta")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--train_chars", type=int, default=None, help="obetnij dane treningowe")
a = p.parse_args()

download_if_missing()
tok, train, val = load_dataset()
if a.train_chars:
    train = train[: a.train_chars]
V, K, C, H = tok.vocab_size, a.K, a.C, a.H
np.random.seed(0)


def windows(ids):
    """X: (n, K) kontekst, Y: (n,) następny znak."""
    n = len(ids) - K
    X = np.lib.stride_tricks.sliding_window_view(ids[:-1], K)[:n]
    return X, ids[K:]


Xtr, Ytr = windows(train)
Xva, Yva = windows(val)

E = Tensor(np.random.randn(V, C) * 0.1, True)
W1 = Tensor(np.random.randn(K * C, H) * (1 / np.sqrt(K * C)), True)
b1 = Tensor(np.zeros(H), True)
W2 = Tensor(np.random.randn(H, V) * 0.01, True)
b2 = Tensor(np.zeros(V), True)
params = [E, W1, b1, W2, b2]
print(f"parametry: {sum(p.data.size for p in params):,}")
opt = AdamW(params, lr=3e-3, weight_decay=0.01)


def forward(X, Y):
    h = (E[X].reshape(X.shape[0], K * C) @ W1 + b1).tanh()
    return cross_entropy(h @ W2 + b2, Y)


def evaluate(X, Y, n=4096):
    i = np.random.randint(0, len(X), n)
    return float(forward(X[i], Y[i]).data)


for step in range(a.steps + 1):
    if step % 250 == 0:
        print(f"step {step:5d}  train {evaluate(Xtr, Ytr):.3f}  val {evaluate(Xva, Yva):.3f}")
    i = np.random.randint(0, len(Xtr), 64)
    loss = forward(Xtr[i], Ytr[i])
    opt.zero_grad(); loss.backward()
    opt.lr = 3e-3 if step < a.steps * 0.7 else 1e-3
    opt.step()

# generowanie
ctx = [tok.stoi["\n"]] * K
out = []
for _ in range(300):
    h = (E[np.array([ctx])].reshape(1, K * C) @ W1 + b1).tanh()
    logits = (h @ W2 + b2).data[0]
    pr = np.exp(logits - logits.max()); pr /= pr.sum()
    nxt = np.random.choice(V, p=pr)
    ctx = ctx[1:] + [nxt]; out.append(tok.itos[nxt])
print("\n--- próbka z MLP ---\n" + "".join(out))
print("\nPorównaj z bigramem (val ~2.49).  MLP z K=8 powinien zejść do ~2.0-2.1.")
print("Ale: MLP 'skleja' kontekst w jeden wektor - każda pozycja ma swoją wagę, nic nie jest")
print("współdzielone między pozycjami.  Attention naprawia dokładnie to (lekcja 4).")
