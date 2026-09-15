# Lekcja 1 — Tekst → liczby. Bigram i pojęcie "loss"

**Czas:** ~2 h · **Cel:** zamieniasz tekst na liczby (tokenizer), dzielisz dane na treningowe i walidacyjne, budujesz najprostszy możliwy model języka (bigram) **bez żadnego uczenia maszynowego**, rozumiesz, co mierzy *loss*, i generujesz pierwszy tekst.

Wszystko w tej lekcji to czysty NumPy i statystyka. Żadnych gradientów. Ale pojęcia — token, batch, cross-entropy, sampling — zostają z Tobą do końca kursu.

---

## 1. Teoria w 5 minut

**Komputer nie rozumie liter.** Model dostaje liczby. Najprostszy sposób: każdemu unikalnemu znakowi w tekście dajemy numer. `'\n'`→0, `' '`→1, ..., `'a'`→39, ... To jest **tokenizer znakowy** (char-level). Prawdziwe LLM-y używają tokenów-podsłów (BPE, np. "Szek"+"spir"), ale zasada jest ta sama, tylko słownik większy (50k–150k zamiast 65).

**Model językowy odpowiada na jedno pytanie:** "mając poprzednie znaki, jakie jest prawdopodobieństwo każdego z 65 możliwych następnych znaków?"

**Bigram** = model, który patrzy tylko na **jeden** poprzedni znak. Po `q` prawie zawsze jest `u`. Po `T` często `h`. To da się po prostu **policzyć** w tekście: tabelka 65×65 "ile razy po A było B".

**Loss (strata)** = liczba mówiąca "jak bardzo model jest zaskoczony prawdziwym tekstem". Dla każdego znaku bierzemy prawdopodobieństwo, które model przypisał **temu znakowi, który naprawdę wystąpił**, i liczymy `-log(p)`. Jeśli model dał p=1.0 (był pewny i miał rację): `-log(1)=0`, zero zaskoczenia. Jeśli dał p=0.01: `-log(0.01)=4.6`, duże zaskoczenie. Średnia z tego po całym tekście to **cross-entropy**. Ta sama miara będzie w GPT.

Punkt odniesienia: model, który zgaduje losowo, daje każdemu z 65 znaków p=1/65, więc loss = `-log(1/65) = ln(65) = 4.17`. **Każdy model musi być poniżej 4.17, inaczej jest bezużyteczny.**

## 2. Budujemy krok po kroku

Utwórz `moj/data.py`. Będziesz go importować w każdej kolejnej lekcji.

### Krok 2.1 — wczytaj tekst i zbuduj słownik

```python
import numpy as np

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))        # lista unikalnych znaków, posortowana -> stały numer
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}   # string -> int
itos = {i: c for i, c in enumerate(chars)}   # int -> string

print(vocab_size, repr("".join(chars)))
```

> ✅ **Sprawdź:** `65 "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"`
> To cały "język" modelu: 65 symboli.

### Krok 2.2 — encode / decode

```python
def encode(s):
    return np.array([stoi[c] for c in s], dtype=np.int64)

def decode(ids):
    return "".join(itos[int(i)] for i in ids)

print(encode("Hello"))
print(decode(encode("Hello")))
```

> ✅ **Sprawdź:** `[20 43 50 50 53]` i `Hello`. Liczby to pozycje w `chars`.

Zapakuj to w klasę, bo będzie wygodniej (w rozwiązaniach: `CharTokenizer` w `data.py`):

```python
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
```

### Krok 2.3 — podział train / val

Ostatnie 10% tekstu **odkładamy** i model go nigdy nie zobaczy podczas nauki. To jak egzamin z pytań, których nie było na zajęciach. Jeśli model radzi sobie dobrze na *train*, a źle na *val* — nauczył się na pamięć zamiast zrozumieć. To będzie **najważniejszy wykres w całym kursie**.

```python
def load_dataset(path="data/input.txt", val_frac=0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    n = int(len(ids) * (1 - val_frac))
    return tok, ids[:n], ids[n:]

tok, train, val = load_dataset()
print(len(train), len(val))
```

> ✅ **Sprawdź:** `1003854 111540`

### Krok 2.4 — bigram przez liczenie

Nowy plik `moj/01_bigram.py`:

```python
import numpy as np
from data import load_dataset

tok, train, val = load_dataset()
V = tok.vocab_size

N = np.zeros((V, V))                         # N[a, b] = ile razy po znaku a był znak b
np.add.at(N, (train[:-1], train[1:]), 1)     # train[:-1] = "poprzedni", train[1:] = "następny"
print(N[tok.stoi["q"], tok.stoi["u"]], N[tok.stoi["q"]].sum())
```

`np.add.at(N, (rows, cols), 1)` dodaje 1 w każdej parze (poprzedni, następny). Jedna linijka zastępuje pętlę po milionie znaków.

> ✅ **Sprawdź:** `563.0 563.0` — po `q` **zawsze** jest `u` (563 z 563 razy).

### Krok 2.5 — z liczników na prawdopodobieństwa

```python
P = (N + 1) / (N + 1).sum(axis=1, keepdims=True)
print(P[tok.stoi["q"]].sum())                # każdy wiersz sumuje się do 1
```

`+1` to **smoothing**: para, która nigdy nie wystąpiła, dostaje maleńkie prawdopodobieństwo zamiast zera. Bez tego `log(0) = -inf` zepsuje loss.

### Krok 2.6 — loss (cross-entropy)

```python
def loss(ids):
    p = P[ids[:-1], ids[1:]]      # prawdopodobieństwo, jakie model dał PRAWDZIWEMU następnemu znakowi
    return -np.log(p).mean()

print("losowo :", np.log(V))
print("train  :", loss(train))
print("val    :", loss(val))
print("perplexity val:", np.exp(loss(val)))
```

> ✅ **Sprawdź:** `losowo: 4.174`, `train: 2.455`, `val: 2.482`, `perplexity: ~12`.

Perplexity = `exp(loss)` = "między iloma znakami model realnie się waha". Losowo: 65. Bigram: 12. GPT z lekcji 5 zejdzie do ~5.

Zwróć uwagę: val jest **odrobinę** gorszy niż train (2.48 vs 2.46). To normalne — model liczył statystyki na train. Duża różnica byłaby alarmem.

### Krok 2.7 — generowanie tekstu (sampling)

Model generuje tekst w pętli: weź ostatni znak → spójrz na jego wiersz w `P` → **wylosuj** następny znak zgodnie z tymi prawdopodobieństwami → doklej → powtórz.

```python
rng = np.random.default_rng(0)
i = tok.stoi["\n"]
out = []
for _ in range(300):
    i = rng.choice(V, p=P[i])
    out.append(tok.itos[i])
print("".join(out))
```

> ✅ **Sprawdź:** bełkot, który *brzmi* trochę jak angielski: `Whan ashe: ayarin se! Wine, ges o tilo g is...`. Są spacje w sensownych miejscach, `q` po którym jest `u`, dwukropki po wielkich literach. Model widzi tylko 1 znak wstecz, więc nie umie zrobić słowa. **Zapamiętaj ten bełkot** — w lekcji 5 porównasz.

Dlaczego losujemy, a nie bierzemy zawsze najbardziej prawdopodobny znak? Spróbuj: zamień `rng.choice(V, p=P[i])` na `np.argmax(P[i])`. Zobaczysz `e e e e e...` albo pętlę `the the the`. Deterministyczny wybór wpada w cykle. To samo dotyczy ChatGPT — dlatego jest parametr "temperature" (lekcja 6).

## 3. Batche — przygotowanie na lekcje 2–5

Modele uczą się na **losowych kawałkach** tekstu, nie na całości naraz. Kawałek = `block_size` znaków (np. 64). Bierzemy ich kilka naraz (`batch_size`, np. 32) — procesor lepiej wykorzystuje macierze niż pojedyncze wektory.

Dopisz do `moj/data.py`:

```python
def get_batch(ids, block_size, batch_size, rng=np.random):
    ix = rng.randint(0, len(ids) - block_size - 1, size=batch_size)   # losowe pozycje startowe
    x = np.stack([ids[i:i + block_size] for i in ix])                # wejście
    y = np.stack([ids[i + 1:i + 1 + block_size] for i in ix])        # cel = wejście przesunięte o 1
    return x, y

x, y = get_batch(train, 8, 2)
print(tok.decode(x[0]), "->", tok.decode(y[0]))
```

> ✅ **Sprawdź:** np. `efore we -> fore we ` — `y` to `x` przesunięte o jeden znak. Na pozycji `t` model ma z `x[:t+1]` przewidzieć `y[t]`. Jeden kawałek 64 znaków = 64 zadania treningowe naraz.

## 4. Ćwiczenia

1. **(obowiązkowe)** Wypisz 5 najbardziej prawdopodobnych znaków po `' '` (spacja), po `'\n'` i po `'T'`. Czy wyniki mają sens? (Podpowiedź: `np.argsort(P[i])[::-1][:5]`.)
2. Zrób **trigram**: tabela 65×65×65 (poprzednie *dwa* znaki). Policz loss na val. Powinien być niższy niż bigramu (~2.1?). Ile pamięci zajmuje tabela? Ile zajęłaby dla 8 znaków wstecz? (65⁸ · 8 bajtów — policz. To jest powód, dla którego potrzebujemy sieci neuronowych zamiast tabelek.)
3. Bez `+1` smoothingu policz loss na val. Co się dzieje i dlaczego?
4. Zamień `rng.choice` na `argmax`. Co generuje model?

## 5. Najczęstsze błędy

- `KeyError` w `encode`: tekst zawiera znak spoza słownika (np. polskie litery, gdy słownik budowałeś na Szekspirze). Słownik musi być zbudowany z tego samego tekstu.
- Loss `nan` lub `inf`: gdzieś `log(0)`. Sprawdź smoothing.
- `val` znacznie gorszy niż `train` już w bigramie: sprawdź, czy przypadkiem nie policzyłeś `N` na całym tekście, a nie na `train`.

## Co zapamiętać

- Tokenizer: tekst ↔ liczby. Słownik = 65 znaków. `vocab_size` pojawi się w każdej warstwie wyjściowej modelu.
- Loss = średnie `-log p(prawdziwy następny znak)`. 4.17 = zgadywanie, 2.48 = bigram. Niżej = lepiej.
- Train/val: val to egzamin z nieznanych pytań. Rozjazd train/val = model uczy się na pamięć.
- Generowanie = losowanie z rozkładu, w pętli, znak po znaku.

➡️ [Lekcja 2 — Jak model się uczy: gradient i własny autograd](02_autograd.md)
