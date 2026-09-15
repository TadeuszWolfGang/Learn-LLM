# Lekcja 5 — Składamy GPT i trenujemy

**Czas:** 3–4 h · **Cel:** dokładasz do attention trzy brakujące elementy (LayerNorm, residual, FFN), składasz pełny blok transformera, dodajesz positional embeddings, piszesz pętlę treningową z warmupem, clippingiem, checkpointami i logiem, **trenujesz mikro-GPT na iPadzie** i rysujesz krzywą train/val.

Po tej lekcji masz od zera napisany, działający GPT. ~110 000 parametrów, ~300 linii własnego kodu.

---

## 1. Teoria w 10 minut: czego brakuje attention

Sam attention to "mieszanie informacji między tokenami". Blok transformera dodaje do tego trzy rzeczy:

**FFN / MLP (feed-forward).** Po attention każdy token *osobno* przechodzi przez małą sieć: `C → 4C → GELU → C`. Attention zbiera kontekst; MLP go "przetwarza". W dużych modelach tu siedzi ~2/3 parametrów i tu "przechowywana jest wiedza".

**Residual (połączenie skrótowe).** Zamiast `x = f(x)` robimy `x = x + f(x)`. Każda warstwa **dopisuje poprawkę** do tego, co już jest, zamiast zastępować. Dzięki temu gradient ma "autostradę" prosto od loss do embeddingu (pochodna `x + f(x)` po `x` to `1 + ...`, nigdy nie znika). Bez residuali sieci głębsze niż kilka warstw **nie da się wytrenować**. Zobaczysz to w lekcji 6.

**LayerNorm.** Przed każdą pod-warstwą normalizujemy wektor każdego tokena: średnia 0, odchylenie 1, potem skalowanie wyuczonymi `gamma`, `beta`. Utrzymuje aktywacje w stałej skali niezależnie od głębokości. Bez tego trening jest kapryśny (działa/nie działa zależnie od `lr` i seeda).

**Positional embedding.** Z lekcji 4: attention nie zna kolejności. Dodajemy do embeddingu tokena drugi wyuczony wektor — **embedding pozycji** (`pos_emb`, macierz `(block_size, C)`). Token `'a'` na pozycji 3 = `tok_emb['a'] + pos_emb[3]`.

Blok (tzw. pre-LN, jak GPT-2):

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Cały model:

```
idx (B,T) → tok_emb[idx] + pos_emb[0..T] → Block × n_layer → LayerNorm → Linear(C→65) → logity (B,T,65)
```

## 2. Budujemy krok po kroku

### Krok 2.1 — GELU i LayerNorm w `moj/autograd.py`

Jeśli zrobiłeś ćwiczenia 2–3 z lekcji 2, masz je. Jeśli nie — teraz. Wzory:

```python
    def gelu(self):
        """GELU (wersja tanh, jak w GPT-2): x * Φ(x), gładki ReLU."""
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        x2 = x * x
        t = np.tanh(c * (x + 0.044715 * x2 * x))
        out = Tensor(0.5 * x * (1 + t), True, (self,), "gelu")
        def _backward(g):
            dinner = c * (1 + 3 * 0.044715 * x2)
            dx = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
            self._acc(g * dx)
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * rstd                                  # znormalizowane
        out = Tensor(xhat * gamma.data + beta.data, True, (self, gamma, beta), "ln")
        def _backward(g):
            gamma._acc(_unbroadcast(g * xhat, gamma.shape))
            beta._acc(_unbroadcast(g, beta.shape))
            dxhat = g * gamma.data
            dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                         - xhat * (dxhat * xhat).mean(-1, keepdims=True))
            self._acc(dx)
        out._backward = _backward
        return out

    def dropout(self, p, training=True):
        """Losowo zeruje ułamek p aktywacji (tylko w treningu). Regularyzacja - lekcja 6."""
        if not training or p == 0.0:
            return self
        mask = (np.random.rand(*self.shape) >= p) / (1.0 - p)
        return self * mask.astype(self.data.dtype)
```

Backward LayerNorm to najbrzydszy wzór w kursie. Nie musisz go wyprowadzać — musisz go **sprawdzić**. Dopisz do gradchecka:

```python
g_ = Tensor(np.random.randn(4), True); b_ = Tensor(np.random.randn(4), True)
xl = Tensor(np.random.randn(3, 4), True)
def L4(): return float((xl.layernorm(g_, b_).gelu() ** 2).sum().data)
l4 = (xl.layernorm(g_, b_).gelu() ** 2).sum(); l4.backward()
check("ln x", xl, L4); check("ln gamma", g_, L4); check("ln beta", b_, L4)
```

> ✅ **Sprawdź:** trzy `OK`.

### Krok 2.2 — LayerNorm, MLP i Block w `moj/model.py`

```python
class LayerNorm:
    def __init__(self, n):
        self.g = Tensor(np.ones(n, dtype=np.float32), requires_grad=True)
        self.b = Tensor(np.zeros(n, dtype=np.float32), requires_grad=True)
    def __call__(self, x):
        return x.layernorm(self.g, self.b)
    def params(self):
        return [self.g, self.b]

class MLP:
    def __init__(self, n_embd, dropout=0.0):
        self.fc = Linear(n_embd, 4 * n_embd)       # rozszerz 4x
        self.proj = Linear(4 * n_embd, n_embd)     # zwęź z powrotem
        self.dropout = dropout
    def __call__(self, x, training=False):
        return self.proj(self.fc(x).gelu()).dropout(self.dropout, training)
    def params(self):
        return self.fc.params() + self.proj.params()

class Block:
    def __init__(self, cfg):
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)
    def __call__(self, x, training=False):
        x = x + self.attn(self.ln1(x), training)    # residual!  x + f(x)
        x = x + self.mlp(self.ln2(x), training)
        return x
    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()
```

Porównaj `Block.__call__` z równaniem z sekcji 1. To dosłownie to samo.

### Krok 2.3 — konfiguracja i cały model

```python
class GPTConfig:
    def __init__(self, vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.n_embd, self.dropout = n_layer, n_head, n_embd, dropout

class GPT:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tok_emb = Tensor(np.random.randn(cfg.vocab_size, cfg.n_embd).astype(np.float32) * 0.02, True)
        self.pos_emb = Tensor(np.random.randn(cfg.block_size, cfg.n_embd).astype(np.float32) * 0.02, True)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        self.head = Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def params(self):
        ps = [self.tok_emb, self.pos_emb]
        for b in self.blocks:
            ps += b.params()
        return ps + self.ln_f.params() + self.head.params()

    def n_params(self):
        return sum(p.data.size for p in self.params())

    def __call__(self, idx, targets=None, training=False):
        B, T = idx.shape
        x = self.tok_emb[idx] + self.pos_emb[np.arange(T)]     # (B,T,C) + (T,C) -> broadcast po B
        for b in self.blocks:
            x = b(x, training)
        logits = self.head(self.ln_f(x))                         # (B, T, V)
        loss = None
        if targets is not None:
            loss = cross_entropy(logits.reshape(B * T, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss
```

(`from autograd import Tensor, cross_entropy` na górze pliku.)

> ✅ **Sprawdź:**
> ```python
> from data import load_dataset, get_batch
> tok, train, val = load_dataset()
> m = GPT(GPTConfig(tok.vocab_size))
> print(m.n_params())                  # 112512
> x, y = get_batch(train, 64, 4)
> logits, loss = m(x, y)
> print(logits.shape, float(loss.data))   # (4, 64, 65)  ~4.17
> ```
> Loss ~4.17 na starcie = model zgaduje losowo = wagi są małe, jak trzeba. Jeśli 5+ — inicjalizacja za duża.

**Gradcheck całego modelu** (ostatni raz; potem możesz ufać kodowi). Dopisz do `moj/02_gradcheck.py`:

```python
from model import GPT, GPTConfig
np.random.seed(0)
m = GPT(GPTConfig(vocab_size=7, block_size=5, n_layer=1, n_head=2, n_embd=8))
for p in m.params():
    p.data = p.data.astype(np.float64)
    if p.data.ndim >= 2: p.data = np.random.randn(*p.data.shape) * 0.5
for blk in m.blocks: blk.attn.mask = blk.attn.mask.astype(np.float64)
xg = np.random.randint(0, 7, size=(2, 5)); yg = np.random.randint(0, 7, size=(2, 5))
def L5(): return float(m(xg, yg)[1].data)
m(xg, yg)[1].backward()
for i, p in enumerate(m.params()):
    check(f"gpt p{i}", p, L5)
```

> ✅ **Sprawdź:** 17 linii `OK`. Gratulacje — masz poprawny transformer z własnym autogradem.

### Krok 2.4 — generowanie i checkpoint (metody w klasie `GPT`)

```python
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        idx = np.array(idx, dtype=np.int64).reshape(1, -1)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]               # model widzi max block_size znaków
            logits, _ = self(ctx)
            logits = logits.data[0, -1] / max(temperature, 1e-6)   # tylko ostatnia pozycja
            if top_k is not None:
                kth = np.sort(logits)[-top_k]
                logits = np.where(logits < kth, -np.inf, logits)
            p = np.exp(logits - logits.max()); p /= p.sum()
            nxt = np.random.choice(len(p), p=p)
            idx = np.concatenate([idx, [[nxt]]], axis=1)
        return idx[0]

    def save(self, path):
        arrs = {f"p{i}": p.data for i, p in enumerate(self.params())}
        np.savez(path, cfg=np.array([repr(self.cfg.__dict__)]), **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = GPTConfig(**eval(str(z["cfg"][0])))
        m = cls(cfg)
        for i, p in enumerate(m.params()):
            p.data[...] = z[f"p{i}"]
        return m
```

Checkpoint = wszystkie wagi w jednym pliku `.npz`. Na iPadzie to ważne: jeśli system ubije proces, wznawiasz zamiast zaczynać od zera.

### Krok 2.5 — clipping gradientu i harmonogram lr (`moj/autograd.py`)

Dwie rzeczy, bez których mikro-transformer potrafi "nie ruszyć" albo wybuchnąć:

```python
def clip_grad_norm(params, max_norm=1.0):
    """Jeśli łączna długość wektora gradientów > max_norm, skróć proporcjonalnie."""
    total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in params if p.grad is not None))
    if total > max_norm:
        s = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= s
    return total
```

Harmonogram lr (w skrypcie treningowym): **warmup** — pierwsze 100 kroków lr rośnie liniowo od 0 (Adam na starcie ma złe statystyki `m`, `v`; duży krok od razu psuje wagi), potem **cosinus** w dół do `min_lr` (małe kroki na końcu = precyzyjne "dostrojenie").

### Krok 2.6 — pętla treningowa `moj/05_train.py`

```python
import os, time
import numpy as np
from data import load_dataset, get_batch
from model import GPT, GPTConfig
from autograd import AdamW, clip_grad_norm

STEPS, BATCH, LR, MIN_LR, WARMUP = 2000, 32, 3e-3, 3e-4, 100
np.random.seed(1337)
os.makedirs("out", exist_ok=True)

tok, train, val = load_dataset()
model = GPT(GPTConfig(tok.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64))
params = model.params()
opt = AdamW(params, lr=LR, weight_decay=0.1)
print("parametry:", model.n_params())

def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + np.cos(np.pi * t))

def estimate_loss(ids, n=10):
    return float(np.mean([float(model(*get_batch(ids, 64, BATCH))[1].data) for _ in range(n)]))

log = open("out/log.csv", "w"); log.write("step,train,val\n")
t0 = time.time(); best = 9e9
for step in range(STEPS + 1):
    if step % 100 == 0:
        tr, va = estimate_loss(train), estimate_loss(val)
        print(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | lr {lr_at(step):.1e} | {time.time()-t0:5.0f}s")
        log.write(f"{step},{tr:.4f},{va:.4f}\n"); log.flush()
        model.save("out/ckpt.npz")                      # zawsze ostatni
        if va < best:
            best = va; model.save("out/best.npz")       # najlepszy na val
    x, y = get_batch(train, 64, BATCH)
    _, loss = model(x, y, training=True)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm(params, 1.0)
    opt.lr = lr_at(step)
    opt.step()

print(tok.decode(model.generate(tok.encode("\n"), 300, temperature=0.8)))
```

Uruchom: `python3 moj/05_train.py` (z folderu repo, żeby `data/input.txt` był widoczny). Podłącz zasilanie, nie zmieniaj aplikacji.

> ✅ **Sprawdź** (2–8 min zależnie od iPada):
> ```
> step     0 | train 4.199 | val 4.199
> step   500 | train 2.04  | val 2.13
> step  1000 | train 1.81  | val 1.94
> step  2000 | train 1.60  | val 1.76
> ```
> Val **1.75** — bigram 2.48, MLP 2.1. A próbka tekstu wygląda tak:
> ```
> MORIZEL:
> He findom hate they, plare that I cunjuster of partle
> Upon thee have than vice to the father;
> ```
> Zmyślone słowa, ale **struktura sztuki**: imiona postaci wielkimi literami z dwukropkiem, wersy, interpunkcja, wielka litera po kropce, angielska morfologia (`-tion`, `-ed`, `the`). To, czego bigram i MLP nie umiały. I tylko 2000 kroków.

### Krok 2.7 — wykres

`moj/plot_loss.py`:

```python
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.genfromtxt("out/log.csv", delimiter=",", names=True)
plt.plot(d["step"], d["train"], label="train"); plt.plot(d["step"], d["val"], "--", label="val")
plt.axhline(np.log(65), ls=":", c="gray", label="zgadywanie 4.17")
plt.axhline(2.48, ls=":", c="orange", label="bigram 2.48")
plt.xlabel("krok"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("out/loss.png", dpi=130)
```

`open out/loss.png`. Obie krzywe spadają, val lekko nad train, odstęp powoli rośnie pod koniec. To jest zdrowy trening. W lekcji 6 zobaczysz niezdrowy.

## 3. Wznawianie po ubiciu procesu

Zamień `model = GPT(GPTConfig(...))` na `model = GPT.load("out/ckpt.npz")`, ustaw `STEPS` na pozostałą liczbę kroków i uruchom ponownie. (W rozwiązaniach: `--resume out/ckpt.npz`.) Stan Adama (`m`, `v`) nie jest zapisany — pierwsze 100 kroków po wznowieniu będzie odrobinę gorsze, warmup to maskuje.

## 4. Ćwiczenia

1. **(obowiązkowe)** Wygeneruj 5 próbek z `out/best.npz` z promptem `"ROMEO:"`. Zapisz najlepszą do notatek. Porównaj z bigramem z lekcji 1.
2. Policz na kartce liczbę parametrów: `tok_emb` 65·64, `pos_emb` 64·64, na blok: `qkv` 64·192+192, `proj` 64·64+64, `fc` 64·256+256, `proj` 256·64+64, 2×LN 2·64·2, `ln_f` 128, `head` 64·65. Sprawdź, czy wychodzi 112 512.
3. Zmierz czas kroku dla `n_embd` 32, 64, 128 (10 kroków każdy). Jak skaluje się czas? Ile parametrów ma każdy?
4. Wytrenuj większy model: `n_layer=4, n_embd=128, STEPS=3000` (15–40 min). Jaki val? (Powinien zejść poniżej 1.6.) Czy tekst jest lepszy "na oko"?
5. Ustaw `WARMUP=0` i `LR=1e-2`. Co się dzieje? A z `clip_grad_norm` wyłączonym?

## 5. Najczęstsze błędy

- Loss stoi na 4.17 przez 200+ kroków → sprawdź, czy `opt.lr = lr_at(step)` jest **przed** `opt.step()`, i czy warmup nie jest za długi.
- Loss spada, potem nagle `nan` → brak clippingu albo `lr` za duży. Zmniejsz `LR` do 1e-3.
- `AssertionError` / błąd kształtu w `pos_emb[np.arange(T)]` → `T > block_size`; kontekst nie może być dłuższy niż `pos_emb`.
- Trening 3× wolniejszy niż w lekcji 0 → iPad throttluje (ciepły?) albo inna aplikacja żre CPU. Poczekaj, ostudź.
- Proces zniknął → iPadOS uśpił aplikację. Wznów z checkpointa (sekcja 3).

## Co zapamiętać

- Blok = `x + Attn(LN(x))`, `x + MLP(LN(x))`. Residual = autostrada gradientu. LN = stała skala. MLP = przetwarzanie per token.
- `pos_emb` daje attention informację o kolejności.
- Warmup + cosinus + clipping to standard; bez nich mikro-transformer jest kapryśny.
- Checkpoint co N kroków, log do CSV, wykres train/val. Ten nawyk zostaje na zawsze.
- 2000 kroków, 110k parametrów, iPad: val 1.75 i tekst ze strukturą Szekspira.

➡️ [Lekcja 6 — Eksperymenty: overfitting, ablacje, temperatura](06_eksperymenty.md)
