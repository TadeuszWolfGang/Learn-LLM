"""
LEKCJA 1 - najprostszy model języka: bigram przez liczenie.

Pytanie: "jaki znak jest następny?" odpowiadamy tabelką: ile razy po znaku A
występował znak B.  Zero uczenia gradientowego - sama statystyka.
To jest punkt odniesienia dla wszystkiego, co dalej.

    python3 01_bigram.py
"""
import numpy as np
from data import download_if_missing, load_dataset

download_if_missing()
tok, train, val = load_dataset()
V = tok.vocab_size
print(f"vocab ({V} znaków): {repr(''.join(tok.chars))}\n")

# 1) tabela zliczeń N[a, b] = ile razy po 'a' było 'b'
N = np.zeros((V, V))
np.add.at(N, (train[:-1], train[1:]), 1)

# 2) prawdopodobieństwa: każdy wiersz sumuje się do 1.  +1 = "smoothing" (żeby nie było log(0))
P = (N + 1) / (N + 1).sum(1, keepdims=True)

# 3) loss = średnia -log P(następny | poprzedni).  To jest cross-entropy, ta sama co w GPT.
def loss(ids):
    return -np.log(P[ids[:-1], ids[1:]]).mean()

print(f"loss zgadywania losowego: ln({V}) = {np.log(V):.3f}")
print(f"loss bigram  train: {loss(train):.3f}   val: {loss(val):.3f}")
print("(mniejszy loss = model mniej zaskoczony prawdziwym tekstem;  exp(loss) = 'ile znaków"
      " realnie waha się model' = perplexity)")
print(f"perplexity val: {np.exp(loss(val)):.1f} znaków (z {V})\n")

# 4) najbardziej prawdopodobne następniki kilku znaków
for c in "q e\nT":
    i = tok.stoi[c]
    top = np.argsort(P[i])[::-1][:5]
    print(f"po {repr(c):5s} -> " + ", ".join(f"{repr(tok.itos[j])}:{P[i, j]:.2f}" for j in top))

# 5) generowanie: losuj następny znak z wiersza P, doklej, powtórz
rng = np.random.default_rng(0)
i = tok.stoi["\n"]
out = []
for _ in range(300):
    i = rng.choice(V, p=P[i])
    out.append(tok.itos[i])
print("\n--- próbka z bigramu (brzmi 'trochę jak angielski', bo widzi tylko 1 znak wstecz) ---")
print("".join(out))
