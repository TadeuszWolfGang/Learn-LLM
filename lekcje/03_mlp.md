# Lekcja 3 — Embedding i sieć neuronowa z kontekstem (MLP)

**Czas:** ~2 h · **Cel:** budujesz pierwszą prawdziwą sieć neuronową do języka: model patrzy na **8 poprzednich znaków**, ma embedding i warstwę ukrytą. Poznajesz Adama. Widzisz overfitting na własne oczy. I rozumiesz, **czego MLP nie umie** — bo to jest powód istnienia attention.

To jest model z pracy Bengio (2003) — "pradziadek" GPT.

---

## 1. Teoria w 10 minut

### Problem z tabelką

Bigram widzi 1 znak. W ćwiczeniu 2 z lekcji 1 policzyłeś: tabelka na 8 znaków wstecz miałaby 65⁸ ≈ 3·10¹⁴ komórek. Nie ma tyle pamięci i nie ma tyle tekstu, żeby ją wypełnić. Potrzebujemy modelu, który **uogólnia**: widząc `"the ca"`, wykorzysta to, czego nauczył się na `"the do"`, bo oba konteksty są "podobne".

### Embedding: znak jako wektor

Zamiast numeru 39 dla `'a'`, dajemy każdemu znakowi **wektor** kilkunastu liczb (np. 16), których model się **uczy**. Znaki występujące w podobnych kontekstach dostają podobne wektory (samogłoski blisko siebie, wielkie litery blisko siebie). To macierz `E` rozmiaru `(65, 16)`; pobranie wiersza `E[idx]` to operacja `gather` z lekcji 2.

### MLP: sklej kontekst, przepuść przez nieliniowość

```
8 znaków  →  8 wektorów po 16  →  sklej w jeden wektor 128  →  W1 (128→256), tanh  →  W2 (256→65)  →  logity
```

`tanh` (albo relu/gelu) to **nieliniowość**. Bez niej dwie macierze `W1·W2` to jedna macierz — sieć byłaby liniowa, czyli równie głupia jak bigram na 8 znakach. Nieliniowość pozwala reprezentować "jeśli 3 znaki temu była spacja I ostatni znak to `t`, to pewnie `h`".

### Adam: mądrzejszy SGD

SGD robi ten sam krok `lr` dla każdej wagi. Ale jedne wagi dostają gradient 1000 razy większy niż inne (np. embedding częstej spacji vs rzadkiego `$`). Adam normalizuje krok każdej wagi przez jej "typową" wielkość gradientu (średnie ruchome `m` i `v`). Efekt: mniej dłubania przy `lr`, szybsza zbieżność. Wszystkie LLM-y trenuje się Adamem (wariant AdamW).

## 2. Budujemy krok po kroku

### Krok 2.1 — AdamW w `moj/autograd.py`

Dopisz pod `SGD`:

```python
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]   # średnia ruchoma gradientu
        self.v = [np.zeros_like(p.data) for p in self.params]   # średnia ruchoma gradientu²
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)      # korekta na start (m, v zaczynają od 0)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            if self.wd and p.data.ndim >= 2:                  # weight decay tylko dla macierzy
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
```

`weight_decay` delikatnie "ściąga" wagi do zera co krok — to regularyzacja: utrudnia modelowi zapamiętywanie danych na pamięć. Zobaczysz efekt w lekcji 6.

### Krok 2.2 — dane w oknach

Plik `moj/03_mlp.py`. Zamiast losowych batchy z `get_batch`, tu wygodniej przygotować wszystkie okna naraz:

```python
import numpy as np
from autograd import Tensor, cross_entropy, AdamW
from data import load_dataset

tok, train, val = load_dataset()
V, K, C, H = tok.vocab_size, 8, 16, 256    # K znaków kontekstu, C wymiar embeddingu, H warstwa ukryta
np.random.seed(0)

def windows(ids):
    n = len(ids) - K
    X = np.lib.stride_tricks.sliding_window_view(ids[:-1], K)[:n]   # (n, K): każde okno K znaków
    return X, ids[K:]                                                 # Y: znak po oknie

Xtr, Ytr = windows(train)
Xva, Yva = windows(val)
print(Xtr.shape, repr(tok.decode(Xtr[100])), "->", repr(tok.itos[int(Ytr[100])]))
```

> ✅ **Sprawdź:** `(1003846, 8) '...8 znaków...' -> 'następny'`. `sliding_window_view` nie kopiuje danych — milion okien zajmuje tyle, co sam tekst.

### Krok 2.3 — parametry i forward

```python
E  = Tensor(np.random.randn(V, C) * 0.1, True)                        # embedding 65 x 16
W1 = Tensor(np.random.randn(K * C, H) * (1 / np.sqrt(K * C)), True)   # 128 x 256
b1 = Tensor(np.zeros(H), True)
W2 = Tensor(np.random.randn(H, V) * 0.01, True)                       # 256 x 65
b2 = Tensor(np.zeros(V), True)
params = [E, W1, b1, W2, b2]
print("parametry:", sum(p.data.size for p in params))

def forward(X, Y):
    emb = E[X]                                               # (B, K, C): wektor dla każdego z K znaków
    h = (emb.reshape(X.shape[0], K * C) @ W1 + b1).tanh()    # (B, H)
    logits = h @ W2 + b2                                     # (B, V)
    return cross_entropy(logits, Y)
```

> ✅ **Sprawdź:** `parametry: 50769`. Inicjalizacja `1/sqrt(wejście)` (tzw. Xavier) utrzymuje aktywacje w rozsądnej skali — bez tego `tanh` nasyca się do ±1 i gradient umiera. `W2 * 0.01` sprawia, że na starcie logity są ~0, czyli loss startuje ~ln(65)=4.17. Sprawdź: `print(float(forward(Xtr[:64], Ytr[:64]).data))`.

### Krok 2.4 — pętla treningowa z oceną train/val

```python
opt = AdamW(params, lr=3e-3, weight_decay=0.01)

def evaluate(X, Y, n=4096):
    i = np.random.randint(0, len(X), n)
    return float(forward(X[i], Y[i]).data)

for step in range(3001):
    if step % 250 == 0:
        print(f"step {step:5d}  train {evaluate(Xtr, Ytr):.3f}  val {evaluate(Xva, Yva):.3f}")
    i = np.random.randint(0, len(Xtr), 64)       # batch: 64 losowych okien
    loss = forward(Xtr[i], Ytr[i])
    opt.zero_grad()
    loss.backward()
    opt.step()
```

> ✅ **Sprawdź** (~2–4 min): train i val spadają razem: 4.17 → ~2.5 (po 250) → ~2.35 (po 1000) → **~2.2–2.25 po 3000** (dalej spada powoli; przy 10 000 kroków ~2.1). Val i train idą łeb w łeb. Bigram miał 2.48. Kontekst 8 znaków + nieliniowość = wyraźny zysk.

### Krok 2.5 — generowanie

```python
ctx = [tok.stoi["\n"]] * K
out = []
for _ in range(300):
    h = (E[np.array([ctx])].reshape(1, K * C) @ W1 + b1).tanh()
    logits = (h @ W2 + b2).data[0]
    p = np.exp(logits - logits.max()); p /= p.sum()      # softmax "ręcznie" (nie potrzebujemy gradientu)
    nxt = np.random.choice(V, p=p)
    ctx = ctx[1:] + [nxt]
    out.append(tok.itos[nxt])
print("".join(out))
```

> ✅ **Sprawdź:** pojawiają się **prawdziwe krótkie słowa** (`the`, `and`, `you`, `my`), struktura `IMIĘ:\n`. Dłuższe słowa nadal zmyślone. Porównaj z bełkotem bigramu z lekcji 1.

## 3. Overfitting na własne oczy

Najważniejszy eksperyment lekcji. Obetnij dane treningowe do 5000 znaków (po `load_dataset()` dopisz `train = train[:5000]`) i uruchom ponownie.

> ✅ **Sprawdź:** train spada do ~1.5 i dalej, **val spada do ~2.9 po 250 krokach, a potem ROŚNIE** (3.1, 3.2, 3.5, …). Model nauczył się 5000 znaków na pamięć. Na egzaminie z nieznanego tekstu jest coraz gorszy, mimo że na "zajęciach" coraz lepszy.

To jest **overfitting**. Rozpoznajesz go po jednym: **rozjazd train i val**. Wszystkie techniki regularyzacji (weight decay, dropout, więcej danych, mniejszy model, wcześniejsze zatrzymanie) służą jednemu: zbliżyć val do train.

Przywróć pełne dane.

## 4. Czego MLP nie umie (motywacja attention)

Zastanów się, co zrobił `reshape(B, K*C)`: **skleił** 8 wektorów w jeden długi. Waga `W1` ma osobne kolumny dla "znaku na pozycji 1", "na pozycji 2", ... Konsekwencje:

1. **Pozycja jest sztywna.** Wzorzec "po dwukropku i nowej linii idzie wielka litera" model musi nauczyć się **osobno** dla każdej pozycji, na której dwukropek może się znaleźć. Nic nie jest współdzielone.
2. **Kontekst jest sztywny.** K=8. Chcesz 64? `W1` rośnie 8×, a wzorców z pozycjami jest 8× więcej do nauczenia. Chcesz 1000? Nierealne.
3. **Wszystko jest równie ważne.** Model nie może powiedzieć "dla tego znaku liczy się głównie znak 5 pozycji temu, resztę olej". Każda pozycja wchodzi z tą samą siłą.

Attention (lekcja 4) rozwiązuje wszystkie trzy: każda pozycja jest przetwarzana **tymi samymi wagami**, kontekst może być dowolnie długi, a model **sam wybiera**, na które pozycje patrzeć. Dlatego transformer wygrał.

## 5. Ćwiczenia

1. **(obowiązkowe)** Uruchom z `K=3` i `K=16`. Zapisz val po 3000 krokach dla K=3, 8, 16. Czy większy kontekst zawsze pomaga? Ile parametrów ma każdy wariant?
2. Wyłącz `tanh` (zostaw `h = emb.reshape(...) @ W1 + b1`). Jaki jest val? Dlaczego prawie jak bigram?
3. W eksperymencie z overfittingiem ustaw `weight_decay=0.5`. Czy val przestaje rosnąć? Co się dzieje z train?
4. Zbadaj embedding: po treningu policz odległości między wierszami `E.data` (np. `np.linalg.norm(E.data[i]-E.data[j])`). Które znaki są najbliżej `'a'`? A `'A'`? (Podpowiedź: samogłoski, wielkie litery.)
5. Zamień `AdamW` na `SGD(lr=0.5)`. Ile kroków potrzeba, żeby dojść do tego samego loss?

## 6. Najczęstsze błędy

- Loss stoi na 4.17 i nie rusza → `lr` za mały albo gradient nie dociera (sprawdź `E.grad is not None` po `backward()`).
- Loss `nan` → za duży `lr` albo brak `-max` w softmaxie.
- `sliding_window_view` nie istnieje → stary NumPy (< 1.20). Zrób `X = np.stack([ids[i:i+K] for i in range(n)])` (wolniej, ale działa).

## Co zapamiętać

- Embedding = wyuczony wektor na token; podobne tokeny → podobne wektory.
- Nieliniowość między macierzami jest **konieczna**; bez niej sieć = jedna macierz.
- Adam ≈ SGD z automatycznym `lr` per waga. Weight decay = regularyzacja.
- Overfitting = train spada, val rośnie. Jedyny wykres, na który zawsze patrzysz.
- MLP skleja kontekst sztywno pozycjami → attention pozwoli modelowi **wybierać**.

➡️ [Lekcja 4 — Attention na piechotę: Q, K, V](04_attention.md)
