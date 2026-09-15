"""
LEKCJA 4 - attention policzone ręcznie w NumPy (bez autograd, bez treningu).

Jedno równanie:   Attention(Q, K, V) = softmax( Q K^T / sqrt(d) + maska ) V

  Q (query)  - "czego szukam"          (dla każdego tokena)
  K (key)    - "co oferuję"            (dla każdego tokena)
  V (value)  - "co przekażę, jeśli mnie wybiorą"
  Q K^T      - podobieństwo każdej pary (kto na kogo patrzy)
  maska      - token nie może patrzeć w przyszłość (causal)
  softmax    - podobieństwa -> wagi sumujące się do 1
  ... V      - wyjście = ważona średnia wartości

    python3 04_attention_by_hand.py     # drukuje macierze i zapisuje attention.png
"""
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
np.random.seed(3)

tokens = ["the", "cat", "sat", "on", "the", "mat"]
T, d = len(tokens), 4

# 1) każdy token ma wektor (embedding).  Tu losowe; w GPT - wyuczone.
x = np.random.randn(T, d)
# 2) trzy macierze wag (wyuczone w prawdziwym modelu) rzutują x na Q, K, V
Wq, Wk, Wv = (np.random.randn(d, d) for _ in range(3))
Q, K, V = x @ Wq, x @ Wk, x @ Wv
print("Q (T x d) =\n", Q, "\n")

# 3) scores[i, j] = jak bardzo token i "interesuje się" tokenem j
scores = Q @ K.T / np.sqrt(d)
print("scores = Q K^T / sqrt(d)  (wiersz = kto patrzy, kolumna = na kogo)\n", scores, "\n")

# 4) maska przyczynowa: zasłoń j > i (przyszłość).  -inf po softmaxie daje 0.
mask = np.triu(np.ones((T, T)), k=1).astype(bool)
scores_masked = np.where(mask, -np.inf, scores)
print("po masce (przyszłość = -inf)\n", scores_masked, "\n")

# 5) softmax po wierszach
e = np.exp(scores_masked - scores_masked.max(1, keepdims=True))
A = e / e.sum(1, keepdims=True)
print("A = softmax(...)   każdy wiersz sumuje się do 1:", A.sum(1))
for i, t in enumerate(tokens):
    print(f"  {t:4s} patrzy na: " + "  ".join(f"{tokens[j]}={A[i, j]:.2f}" for j in range(i + 1)))

# 6) wyjście = ważona średnia V
out = A @ V
print("\nout = A V  (T x d): nowa reprezentacja każdego tokena, 'wzbogacona' o kontekst\n", out)

# 7) Sprawdzenie intuicji: gdy Q tokena 'sat' celuje w K tokena 'cat', waga rośnie.
print("\n--- eksperyment: ustawiam Q[sat] := K[cat] * 5 ---")
Q2 = Q.copy(); Q2[2] = K[1] * 5
s2 = np.where(mask, -np.inf, Q2 @ K.T / np.sqrt(d))
A2 = np.exp(s2 - s2.max(1, keepdims=True)); A2 /= A2.sum(1, keepdims=True)
print("sat patrzy na:", {tokens[j]: round(float(A2[2, j]), 2) for j in range(3)})
print("=> attention to 'miękkie wyszukiwanie w słowniku': query wybiera key, dostaje value.")

# 8) multi-head = to samo kilka razy równolegle z mniejszym d, wyniki sklejone.
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a_, M, title in [(ax[0], A, "losowe wagi"), (ax[1], A2, "Q[sat] celuje w K[cat]")]:
        a_.imshow(M, cmap="Blues", vmin=0, vmax=1)
        a_.set_xticks(range(T)); a_.set_xticklabels(tokens); a_.set_yticks(range(T)); a_.set_yticklabels(tokens)
        a_.set_title(title); a_.set_xlabel("na kogo patrzy (key)"); a_.set_ylabel("kto patrzy (query)")
    plt.tight_layout(); plt.savefig("attention.png", dpi=130)
    print("\nzapisano attention.png (a-Shell: open attention.png)")
except ImportError:
    pass
