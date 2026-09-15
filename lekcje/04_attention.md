# Lekcja 4 — Attention na piechotę: Q, K, V

**Czas:** 2–3 h · **Cel:** liczysz attention **ręcznie w NumPy** na zdaniu z 6 słów, bez treningu, bez autogradu. Rozumiesz każdą z 5 rzeczy: Q, K, V, maskę i softmax. Potem zapisujesz to jako warstwę na Tensorach z lekcji 2 i **sprawdzasz gradcheckiem**.

Jedno równanie, które jest całym "sekretem" GPT:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d  +  maska ) · V
```

---

## 1. Teoria w 10 minut: attention jako "miękkie wyszukiwanie"

Wyobraź sobie, że każdy token w zdaniu to osoba w pokoju. Każda osoba:

- ma **query (Q)** — karteczkę "czego szukam" (np. `sat` szuka: "kto jest podmiotem?"),
- ma **key (K)** — karteczkę "co oferuję" (np. `cat` oferuje: "jestem rzeczownikiem, podmiotem"),
- ma **value (V)** — "co przekażę, jeśli ktoś mnie wybierze" (informacja o `cat`).

Każda osoba porównuje swoje Q z K **wszystkich** (iloczyn skalarny = "jak bardzo pasujemy"). Softmax zamienia to na wagi sumujące się do 1. Wynik dla osoby = ważona suma V wszystkich. `sat` dostaje głównie informację od `cat`, trochę od `the`, i idzie dalej "wiedząc", kto siedział.

Trzy kluczowe własności (rozwiązują trzy problemy MLP z lekcji 3):

1. **Te same wagi dla każdej pozycji.** Q, K, V liczymy z embeddingu przez trzy macierze `Wq, Wk, Wv` — te same dla tokena na pozycji 1 i 500.
2. **Dowolna długość kontekstu.** Macierz wag `T×T` po prostu rośnie z T. (Kwadratowo — to jest powód, dla którego długi kontekst jest drogi.)
3. **Model sam wybiera, na co patrzeć.** Wagi nie są ustalone — zależą od treści (Q·K).

**Maska przyczynowa (causal):** model uczy się przewidywać *następny* znak, więc token na pozycji `t` nie może patrzeć na pozycje `> t` (to byłoby ściąganie). Zasłaniamy przyszłość, wstawiając `-∞` przed softmaxem → po softmaxie waga 0.

**Skąd `/√d`?** Iloczyn skalarny wektorów długości `d` rośnie z `d`. Bez skalowania softmax "wyostrza się" do 0/1 i gradient umiera. Dzielenie przez `√d` utrzymuje rozsądną skalę.

**Multi-head:** zamiast jednego attention na wektorach długości 64, robimy 4 równoległe na wektorach 16 i sklejamy wyniki. Każda "głowa" może uczyć się innego wzorca (jedna patrzy na poprzedni znak, inna na początek słowa, inna na początek linii). Zobaczysz to w lekcji 6.

## 2. Budujemy krok po kroku — ręcznie w NumPy

Plik `moj/04_attention.py`. Bez autogradu. Tylko liczby.

### Krok 2.1 — tokeny i losowe embeddingi

```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
np.random.seed(3)

tokens = ["the", "cat", "sat", "on", "the", "mat"]
T, d = len(tokens), 4                 # 6 tokenów, wektory 4-wymiarowe (mało, żeby dało się czytać)
x = np.random.randn(T, d)             # embedding każdego tokena (losowy; w GPT wyuczony)
```

### Krok 2.2 — Q, K, V przez trzy macierze

```python
Wq, Wk, Wv = (np.random.randn(d, d) for _ in range(3))    # w GPT: wyuczone
Q, K, V = x @ Wq, x @ Wk, x @ Wv                          # każdy (T, d)
print("Q =\n", Q)
```

Zauważ: **ta sama** `Wq` dla wszystkich 6 tokenów. Jedna macierz `(4,4)` niezależnie od T.

### Krok 2.3 — scores: kto na kogo patrzy

```python
scores = Q @ K.T / np.sqrt(d)         # (T, T): scores[i, j] = jak bardzo token i "interesuje się" tokenem j
print("scores =\n", scores)
```

Wiersz `i` = token, który patrzy (query). Kolumna `j` = token, na który patrzy (key).

### Krok 2.4 — maska: zasłoń przyszłość

```python
mask = np.triu(np.ones((T, T)), k=1).astype(bool)   # True nad przekątną = przyszłość
scores = np.where(mask, -np.inf, scores)
print("po masce =\n", scores)
```

> ✅ **Sprawdź:** nad przekątną same `-inf`. Wiersz 0 (`the`) ma tylko jedną wartość — pierwszy token widzi tylko siebie.

### Krok 2.5 — softmax po wierszach

```python
e = np.exp(scores - scores.max(axis=1, keepdims=True))
A = e / e.sum(axis=1, keepdims=True)
print("sumy wierszy:", A.sum(1))
for i, t in enumerate(tokens):
    print(f"  {t:4s} patrzy na: " + "  ".join(f"{tokens[j]}={A[i, j]:.2f}" for j in range(i + 1)))
```

> ✅ **Sprawdź:** sumy = `[1. 1. 1. 1. 1. 1.]`. Każdy token ma rozkład wag na siebie i poprzedników; przyszłość = 0 (bo `exp(-inf) = 0`).

### Krok 2.6 — wyjście = ważona średnia V

```python
out = A @ V                            # (T, d)
print("out =\n", out)
```

`out[i]` to nowy wektor tokena `i`: mieszanka V wszystkich tokenów, na które patrzył, w proporcjach `A[i]`. Token "wie" teraz coś o kontekście. Tyle. To jest attention.

### Krok 2.7 — eksperyment: sterowanie uwagą

Losowe wagi dają losowe spojrzenia. Pokaż sobie, że Q **wybiera** K:

```python
Q2 = Q.copy()
Q2[2] = K[1] * 5                       # query tokena "sat" := (wzmocniony) key tokena "cat"
s2 = np.where(mask, -np.inf, Q2 @ K.T / np.sqrt(d))
A2 = np.exp(s2 - s2.max(1, keepdims=True)); A2 /= A2.sum(1, keepdims=True)
print("sat patrzy na:", {tokens[j]: round(float(A2[2, j]), 2) for j in range(3)})
```

> ✅ **Sprawdź:** `{'the': 0.1, 'cat': 0.9, 'sat': 0.0}` — `sat` prawie całą uwagę kieruje na `cat`. Trening robi dokładnie to: uczy `Wq`, `Wk` tak, żeby "właściwe" pary miały duży iloczyn.

### Krok 2.8 — narysuj to

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a_, M, title in [(ax[0], A, "losowe wagi"), (ax[1], A2, "Q[sat] celuje w K[cat]")]:
    a_.imshow(M, cmap="Blues", vmin=0, vmax=1)
    a_.set_xticks(range(T)); a_.set_xticklabels(tokens)
    a_.set_yticks(range(T)); a_.set_yticklabels(tokens)
    a_.set_title(title); a_.set_xlabel("na kogo (key)"); a_.set_ylabel("kto (query)")
plt.tight_layout(); plt.savefig("attention.png", dpi=130)
```

W a-Shell: `open attention.png`. W Carnets/Pythonista: zamiast `savefig` daj `plt.show()`. Górny trójkąt biały (maska), prawy obrazek ma ciemny kwadrat na (sat, cat).

## 3. Attention jako warstwa z gradientem

Teraz to samo na `Tensor` z lekcji 2, jako klasa, którą w lekcji 5 wstawisz do GPT. Potrzebujesz z autogradu: `@`, `+`, `*`, `reshape`, `transpose`, `swap_last2`, `softmax`. Dwie nowości:

**(a) Multi-head przez reshape.** Zamiast 3 macierzy `Wq, Wk, Wv` używamy jednej `(C, 3C)` i wynik dzielimy. Dla `n_head` głów: `(B, T, 3C)` → `reshape (B, T, 3, nh, hs)` → `transpose (3, B, nh, T, hs)` → wyciągnij Q, K, V. Każda głowa to niezależny "plasterek" o szerokości `hs = C / nh`.

**(b) Maska jako dodawanie.** Zamiast `np.where(-inf)` dodajemy macierz z `0` (widać) i `-1e9` (zasłonięte). Dodawanie już ma backward; `-1e9` po softmaxie = 0 tak samo dobrze jak `-inf`, a nie psuje gradientu.

Dopisz do `moj/model.py` (nowy plik). Najpierw pomocnicza `Linear` i selektor:

```python
import numpy as np
from autograd import Tensor

class Linear:
    def __init__(self, n_in, n_out, std=0.02, bias=True):
        self.w = Tensor(np.random.randn(n_in, n_out).astype(np.float32) * std, requires_grad=True)
        self.b = Tensor(np.zeros(n_out, dtype=np.float32), requires_grad=True) if bias else None
    def __call__(self, x):
        y = x @ self.w
        return y + self.b if self.b is not None else y
    def params(self):
        return [self.w] + ([self.b] if self.b is not None else [])

def _split3(t):
    """Tensor (3, ...) -> trzy Tensory (...), każdy z własnym backward do właściwego plasterka."""
    outs = []
    for i in range(3):
        o = Tensor(t.data[i], True, (t,), f"sel{i}")
        def _backward(g, i=i):
            gg = np.zeros_like(t.data); gg[i] = g; t._acc(gg)
        o._backward = _backward
        outs.append(o)
    return outs
```

Teraz warstwa (porównaj linijka po linijce z sekcją 2 — to ten sam kod, tylko z osią batch i osią głów):

```python
class CausalSelfAttention:
    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv = Linear(n_embd, 3 * n_embd)     # jedna macierz dla Q, K, V
        self.proj = Linear(n_embd, n_embd)        # po sklejeniu głów: wymieszaj
        tril = np.tril(np.ones((block_size, block_size), dtype=np.float32))
        self.mask = np.where(tril == 1, 0.0, -1e9).astype(np.float32)
        self.last_att = None                      # zapamiętamy wagi do podglądu

    def __call__(self, x, training=False):
        B, T, C = x.shape
        qkv = self.qkv(x)                                        # (B, T, 3C)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.hs).transpose(2, 0, 3, 1, 4)   # (3, B, nh, T, hs)
        q, k, v = _split3(qkv)
        att = (q @ k.swap_last2()) * (1.0 / np.sqrt(self.hs))    # (B, nh, T, T)   = Q Kᵀ / √d
        att = att + self.mask[:T, :T]                            # zasłoń przyszłość
        att = att.softmax(-1)                                    # wagi
        self.last_att = att.data
        y = att @ v                                              # (B, nh, T, hs)  = A V
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)             # sklej głowy z powrotem w C
        return self.proj(y)

    def params(self):
        return self.qkv.params() + self.proj.params()
```

### Gradcheck warstwy (obowiązkowy)

Dopisz do `moj/02_gradcheck.py`:

```python
from model import CausalSelfAttention
np.random.seed(1)
attn = CausalSelfAttention(n_embd=8, n_head=2, block_size=5)
for p in attn.params():
    p.data = np.random.randn(*p.data.shape) * 0.5      # float64 i większe wagi = ostrzejszy test
attn.mask = attn.mask.astype(np.float64)
xin = Tensor(np.random.randn(2, 5, 8), True)
def L3(): return float((attn(xin) ** 2).sum().data)
l3 = (attn(xin) ** 2).sum(); l3.backward()
check("attn x", xin, L3); check("attn qkv.w", attn.qkv.w, L3); check("attn proj.w", attn.proj.w, L3)
```

> ✅ **Sprawdź:** trzy `OK`. Jeśli `attn x` jest OK, a `qkv.w` nie — błąd w `_split3` lub `transpose`. Jeśli wszystko źle — najpewniej `softmax` lub `swap_last2`.

Dodatkowy test maski: `attn.last_att[0, 0]` musi mieć zera nad przekątną. Sprawdź `print(attn.last_att[0, 0].round(2))`.

## 4. Ćwiczenia

1. **(obowiązkowe)** W sekcji 2 usuń maskę (pomiń krok 2.4). Który token teraz patrzy na przyszłość? Dlaczego to byłoby "ściąganie" przy przewidywaniu następnego znaku?
2. Zmień `d` na 64 i usuń `/ np.sqrt(d)`. Wypisz `A`. Co się stało z wagami? (Powinny być prawie same 0 i 1.) To jest problem, który rozwiązuje skalowanie.
3. Policz "ręcznie" (w NumPy, bez klasy) attention z 2 głowami dla `d=4`: podziel Q, K, V na dwie połówki po 2, policz osobno, sklej. Porównaj z wynikiem klasy dla tych samych wag.
4. Ile mnożeń kosztuje `Q @ K.T` dla T=64? A dla T=1024? Dla T=100 000 (kontekst dużych modeli)? Stąd bierze się "długi kontekst jest drogi".
5. Attention **nie wie, w jakiej kolejności są tokeny** (sprawdź: przetasuj wiersze `x` — wynik dla każdego tokena jest ten sam, tylko przetasowany). Jak myślisz, skąd GPT wie, że `cat` jest przed `sat`? (Odpowiedź w lekcji 5: positional embeddings.)

## Co zapamiętać

- Q = czego szukam, K = co oferuję, V = co przekażę. `softmax(QKᵀ/√d)·V` = ważona średnia V wg dopasowania Q do K.
- Maska: token widzi tylko siebie i przeszłość. Bez niej model ściąga.
- Te same `Wq, Wk, Wv` dla każdej pozycji → dowolny kontekst, współdzielone wzorce, model sam wybiera, na co patrzeć.
- Multi-head = kilka małych attention równolegle, sklejone.
- Attention nie zna kolejności — potrzebuje informacji o pozycji.

➡️ [Lekcja 5 — Składamy GPT i trenujemy](05_gpt.md)
