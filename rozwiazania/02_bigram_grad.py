"""
LEKCJA 2 - ten sam bigram, ale wyuczony gradientem (na naszym autograd).

Zamiast liczyć tabelkę, mamy macierz wag W (V x V) = "logity", startujemy od
losowej i minimalizujemy cross-entropy.  Powinno zejść do ~2.45 - dokładnie tam,
gdzie bigram przez liczenie.  To dowód, że loss + gradient robi to samo co statystyka,
tylko że skaluje się na modele, dla których nie da się "policzyć tabelki".

    python3 02_bigram_grad.py
"""
import numpy as np
from autograd import Tensor, cross_entropy, AdamW
from data import download_if_missing, load_dataset, get_batch

download_if_missing()
tok, train, val = load_dataset()
V = tok.vocab_size
np.random.seed(0)

W = Tensor(np.random.randn(V, V) * 0.1, requires_grad=True)   # tok_emb == logity: model 65x65
opt = AdamW([W], lr=0.05)


def forward(x, y):
    logits = W[x.reshape(-1)]           # (N, V): wiersz W dla każdego znaku wejściowego
    return cross_entropy(logits, y)


for step in range(301):
    x, y = get_batch(train, 32, 64)
    loss = forward(x, y)
    if step % 50 == 0:
        xv, yv = get_batch(val, 32, 256)
        print(f"step {step:3d}  train {float(loss.data):.3f}  val {float(forward(xv, yv).data):.3f}")
    opt.zero_grad()
    loss.backward()
    opt.step()

print("\nSprawdź: lekcja 1 dała val ≈ 2.49.  Gradient dochodzi w to samo miejsce.")
print("Kolejna lekcja: dajemy modelowi więcej niż 1 znak kontekstu -> loss spada poniżej bigramu.")
