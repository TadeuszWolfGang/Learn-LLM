# Lekcja 2 — Jak model się uczy: gradient i własny autograd

**Czas:** 4–5 h (najdłuższa i najważniejsza lekcja) · **Cel:** piszesz własny silnik **automatycznego różniczkowania** na macierzach NumPy (~250 linii), testujesz go numerycznie, i trenujesz bigram gradientem — dochodząc do tego samego loss, co przez liczenie. Po tej lekcji wiesz, co robi `loss.backward()` w PyTorchu, bo sam to napisałeś.

To jest ten plik, o którym każdy mówi "magia frameworka". Nie ma magii. Jest reguła łańcuchowa.

---

## 1. Teoria w 10 minut

### Po co gradient?

W lekcji 1 model = tabelka, którą **policzyliśmy**. Dla sieci neuronowej nie ma wzoru "policz tabelkę". Zamiast tego:

1. Startujemy z **losowych** wag.
2. Liczymy loss (jak bardzo model jest zaskoczony).
3. Dla **każdej** wagi pytamy: "gdybym tę wagę odrobinę zwiększył, loss by wzrósł czy zmalał?" — to jest **gradient** (pochodna loss względem wagi).
4. Przesuwamy każdą wagę **w stronę malejącego loss** o mały krok (`lr`, learning rate).
5. Powtarzamy tysiące razy.

To cały "trening". Wszystko, co dalej, to szczegóły, jak policzyć punkt 3 szybko dla 100 000 wag naraz.

### Pochodna "na palcach"

Pochodna funkcji `L(w)` w punkcie `w` to: "o ile zmieni się `L`, jak `w` zmienię o malutkie `h`", podzielone przez `h`:

```
dL/dw ≈ (L(w + h) - L(w - h)) / (2h)
```

To jest **pochodna numeryczna**. Można nią policzyć gradient każdej wagi... ale trzeba policzyć `L` 2× na każdą wagę. Dla 100k wag = 200k przebiegów modelu na **jeden** krok. Za wolno. Ale jest **idealna do testowania**, czy nasz szybki sposób liczy dobrze. Użyjemy jej w kroku 2.6.

### Reguła łańcuchowa = cały autograd

Jeśli `L = f(g(w))`, to `dL/dw = dL/dg · dg/dw`. Pochodna "przez" kilka operacji to iloczyn pochodnych każdej z nich.

Model to długi łańcuch prostych operacji: mnożenie macierzy, dodawanie, `tanh`, softmax... Każda z nich **z osobna** ma prostą pochodną. Autograd:

1. Podczas liczenia w przód (**forward**) każda operacja zapamiętuje, z czego powstała (budujemy **graf**).
2. Na końcu mamy `loss`. Ustawiamy `loss.grad = 1` ("loss zmienia się 1:1 sam ze sobą").
3. Idziemy **wstecz** (**backward**) od loss do wag. Każda operacja bierze gradient swojego wyniku i **rozdaje** go swoim wejściom, mnożąc przez swoją lokalną pochodną.

Gdy dojdziemy do wag, każda ma w `.grad` dokładnie `dL/dw`. Jeden przebieg wstecz = gradienty **wszystkich** wag naraz.

### Dlaczego nie micrograd

Micrograd Karpathy'ego robi dokładnie to, ale dla pojedynczych liczb. Nasz model ma 100k wag, każda operacja na macierzy 32×64×64 to 131k liczb. Micrograd stworzyłby miliony obiektów Pythona na jeden krok. My robimy **to samo, ale jeden obiekt = cała macierz**, a matematykę na macierzy robi NumPy w C. Różnica: godziny vs sekundy.

## 2. Budujemy krok po kroku

Utwórz `moj/autograd.py`. Budujemy klasę `Tensor` operacja po operacji. **Po każdym kroku uruchom test** z ramki.

### Krok 2.1 — szkielet Tensora

```python
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        self.data = data                 # wartości (tablica NumPy)
        self.requires_grad = requires_grad
        self.grad = None                 # tu wyląduje dL/d(data)
        self._backward = lambda g: None  # funkcja rozdająca gradient g dzieciom
        self._prev = tuple(_children)    # z czego powstałem (graf)
        self._op = _op                   # nazwa operacji (do debugowania)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def _acc(self, g):
        """Dodaj gradient. Ten sam tensor może być użyty w kilku miejscach -> gradienty się sumują."""
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype)   # kopia!
        else:
            self.grad += g

    @staticmethod
    def _wrap(x):
        if isinstance(x, Tensor):
            return x
        if isinstance(x, (int, float, np.number)):
            return Tensor(np.asarray(x, dtype=np.float32))   # skalar zawsze float32 (patrz uwaga w 2.5)
        return Tensor(x)
```

Dlaczego `_acc` **dodaje**, a nie przypisuje? Jeśli `w` użyjesz dwa razy (np. `y = w*a + w*b`), gradient przychodzi z dwóch stron i trzeba je zsumować. To jedna z najczęstszych pomyłek przy pisaniu autogradu.

### Krok 2.2 — dodawanie i mnożenie (i broadcasting)

Zanim napiszesz `__add__`, jeden problem: NumPy pozwala dodać wektor `(64,)` do macierzy `(32, 64)` — "rozciąga" go na każdy wiersz (**broadcasting**). Tak dodajemy bias. Ale wtedy gradient biasu musi być **sumą** gradientów ze wszystkich 32 wierszy. Funkcja pomocnicza (poza klasą, na górze pliku):

```python
def _unbroadcast(grad, shape):
    """Zsumuj gradient po osiach, które NumPy rozciągnął przy broadcastingu."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

Teraz operacje (w klasie):

```python
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, True, (self, other), "+")
        def _backward(g):
            # d(a+b)/da = 1, d(a+b)/db = 1  -> gradient przechodzi bez zmian
            self._acc(_unbroadcast(g, self.shape))
            other._acc(_unbroadcast(g, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, True, (self, other), "*")
        def _backward(g):
            # d(a*b)/da = b, d(a*b)/db = a
            self._acc(_unbroadcast(g * other.data, self.shape))
            other._acc(_unbroadcast(g * self.data, other.shape))
        out._backward = _backward
        return out

    def __neg__(self):      return self * -1.0
    def __sub__(self, o):   return self + (-self._wrap(o))
    def __radd__(self, o):  return self + o
    def __rmul__(self, o):  return self * o
    def __rsub__(self, o):  return self._wrap(o) + (-self)
```

**Wzorzec, który powtórzysz dla każdej operacji:**
1. policz wynik w NumPy,
2. opakuj w `Tensor`, podając dzieci,
3. zdefiniuj `_backward`: weź `g`, pomnóż przez lokalną pochodną, oddaj dzieciom przez `_acc`.

### Krok 2.3 — backward: przejście po grafie

```python
    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:                                  # DFS bez rekurencji (patrz uwaga niżej)
            t, done = stack.pop()
            if done:
                topo.append(t)                        # dodaj dopiero, gdy wszystkie dzieci już są
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for c in t._prev:
                if id(c) not in seen:
                    stack.append((c, False))
        self.grad = np.ones_like(self.data)           # dL/dL = 1
        for t in reversed(topo):
            t._backward(t.grad)
```

Pętla układa tensory tak, żeby każdy był **po** swoich dzieciach (sortowanie topologiczne). Potem idziemy od końca: najpierw loss, potem to, z czego loss powstał, i tak dalej — gradient płynie "w dół" do wag.

> ⚠️ **Dlaczego bez rekurencji i dlaczego `_backward(g)` dostaje gradient jako argument.** Micrograd robi to rekurencyjną funkcją wewnętrzną i closure czytającym `out.grad`. Oba tworzą **cykle referencji** (funkcja → komórka closure → funkcja; `out` → closure → `out`). Python zwalnia cykle dopiero przy uruchomieniu garbage collectora, nie od razu — więc cały graf jednego kroku (dziesiątki–setki MB tablic) zostaje w pamięci przez wiele kroków. Na Linuksie to "tylko" 10 GB RAM po 20 krokach. Na iPadzie system zabija aplikację. Dlatego: pętla zamiast rekurencji, gradient jako argument, i w closure nigdy nie używaj `out` (tylko zapamiętane lokalne tablice, np. `y`).

> ✅ **Sprawdź** (plik `moj/test_autograd.py`):
> ```python
> from autograd import Tensor
> a = Tensor(2.0, True); b = Tensor(3.0, True)
> L = a * b + a          # L = ab + a  ->  dL/da = b + 1 = 4,  dL/db = a = 2
> L.backward()
> print(a.grad, b.grad)  # 4.0 2.0
> ```
> Zauważ: `a` użyte dwa razy, gradient = 3 + 1 = 4. Gdyby `_acc` przypisywało zamiast dodawać, dostałbyś 1 albo 3.

### Krok 2.4 — mnożenie macierzy

To jest operacja, która robi 90% pracy w LLM. Dla `C = A @ B`:

```
dA = dC @ B^T
dB = A^T @ dC
```

(Skąd to? `C[i,j] = Σ_k A[i,k]·B[k,j]`, więc `dC[i,j]` wpływa na `A[i,k]` przez `B[k,j]`. Zsumuj po `j` — wychodzi `dC @ B^T`. Nie musisz wyprowadzać, ale sprawdzisz numerycznie w 2.6.)

```python
    def __matmul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, True, (self, other), "@")
        def _backward(g):
            a, b = self.data, other.data
            bt = np.swapaxes(b, -1, -2) if b.ndim > 1 else b
            at = np.swapaxes(a, -1, -2) if a.ndim > 1 else a
            self._acc(_unbroadcast(g @ bt, self.shape))
            other._acc(_unbroadcast(at @ g, other.shape))
        out._backward = _backward
        return out
```

`swapaxes(-1, -2)` zamiast `.T`, bo w attention będziemy mnożyć tensory 4-wymiarowe `(batch, głowy, T, d)` i transponujemy tylko dwie ostatnie osie. `_unbroadcast` załatwia przypadek "wejście 3D `(B,T,C)` razy waga 2D `(C,D)`" — gradient wagi sumuje się po batchu.

### Krok 2.5 — sum, mean, kształty, funkcje elementarne

Wszystkie według tego samego wzorca. Przepisz, czytając komentarz z pochodną. Uwaga do `_wrap` wyżej: w NumPy ≥ 2 skalar typu `np.float64` (np. wynik `1/np.sqrt(d)`) pomnożony przez tablicę float32 daje **float64** — cały model po cichu robi się 2× większy i wolniejszy. Dlatego skalary opakowujemy jako float32.

```python
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), True, (self,), "sum")
        def _backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())   # d(sum)/dx_i = 1 dla każdego i
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), True, (self,), "reshape")
        def _backward(g):
            self._acc(g.reshape(self.shape))             # tylko zmiana kształtu z powrotem
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = tuple(axes)
        out = Tensor(self.data.transpose(axes), True, (self,), "T")
        inv = np.argsort(axes)                                    # permutacja odwrotna
        def _backward(g):
            self._acc(g.transpose(inv))
        out._backward = _backward
        return out

    def swap_last2(self):
        axes = list(range(self.data.ndim)); axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    def __pow__(self, p):
        out = Tensor(self.data ** p, True, (self,), f"**{p}")
        def _backward(g):
            self._acc(g * p * self.data ** (p - 1))       # d(x^p)/dx = p x^(p-1)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        return self * (self._wrap(other) ** -1)

    def exp(self):
        y = np.exp(self.data)
        out = Tensor(y, True, (self,), "exp")
        def _backward(g):
            self._acc(g * y)                              # d(e^x)/dx = e^x   (y, NIE out.data - cykl!)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), True, (self,), "log")
        def _backward(g):
            self._acc(g / self.data)                      # d(ln x)/dx = 1/x
        out._backward = _backward
        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Tensor(y, True, (self,), "tanh")
        def _backward(g):
            self._acc(g * (1 - y * y))                    # d(tanh)/dx = 1 - tanh²
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), True, (self,), "relu")
        def _backward(g):
            self._acc(g * (self.data > 0))                # 1 gdzie x>0, 0 gdzie nie
        out._backward = _backward
        return out
```

### Krok 2.6 — TEST NUMERYCZNY (nie pomijaj!)

Teraz najważniejsze 20 linii w kursie. Sprawdzamy, czy nasz backward zgadza się z pochodną numeryczną z sekcji 1. Plik `moj/02_gradcheck.py`:

```python
import numpy as np
from autograd import Tensor

def numeric_grad(f, x, h=1e-5):
    """Pochodna numeryczna f() względem każdego elementu tablicy x (modyfikuje x w miejscu)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g

def check(name, t, f, tol=1e-4):
    num = numeric_grad(f, t.data)
    err = np.abs(num - t.grad).max() / (np.abs(num).max() + 1e-8)
    print(f"{name:10s} max błąd względny = {err:.1e}  {'OK' if err < tol else 'BŁĄD!'}")

np.random.seed(0)
a = Tensor(np.random.randn(3, 4), True)      # float64 -> pochodna numeryczna jest dokładna
b = Tensor(np.random.randn(4, 5), True)
c = Tensor(np.random.randn(5), True)          # bias -> test broadcastingu

def L():   # skalar: suma po tanh(a@b + c)²  (używa matmul, add z broadcastem, tanh, pow, sum)
    return float((((a @ b) + c).tanh() ** 2).sum().data)

loss = (((a @ b) + c).tanh() ** 2).sum()
loss.backward()
check("a", a, L); check("b", b, L); check("c (bias)", c, L)
```

> ✅ **Sprawdź:** trzy linie z `OK` i błędem rzędu `1e-9`…`1e-11`. Jeśli któryś jest `BŁĄD!`, masz pomyłkę w backward tej operacji, która dotyka tego tensora. Najczęściej: `_unbroadcast` (test `c`) albo transpozycja w matmul (test `b`).

**Od teraz każdą nową operację dopisujesz do tego testu.** To jest Twój "pas bezpieczeństwa" na resztę kursu.

### Krok 2.7 — softmax i cross-entropy (sklejone)

Można je złożyć z `exp`, `sum`, `log`. Ale (a) `exp(1000)` = `inf`, (b) 6 małych operacji jest wolniejsze niż jedna. Więc piszemy je jako **jedne operacje z własnym backward** — dokładnie tak robią frameworki.

**Softmax** zamienia dowolne liczby (logity) w prawdopodobieństwa sumujące się do 1: `p_i = e^{x_i} / Σ e^{x_j}`. Trik stabilności: odjąć maksimum przed `exp` (wynik ten sam, bez `inf`).

```python
    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        y = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(y, True, (self,), "softmax")
        def _backward(g):
            self._acc(y * (g - (g * y).sum(axis=axis, keepdims=True)))   # wzór na pochodną softmaxu
        out._backward = _backward
        return out
```

**Cross-entropy** (funkcja poza klasą): `logits` `(N, V)`, `targets` — `N` numerów prawdziwych znaków. Loss = średnia `-log softmax(logits)[prawdziwy]`. Pochodna ma piękną postać: `softmax - onehot(prawdziwy)`, podzielone przez `N`.

```python
def cross_entropy(logits, targets):
    targets = np.asarray(targets).reshape(-1)
    x = logits.data.reshape(-1, logits.shape[-1])
    x = x - x.max(-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(-1, keepdims=True))     # log softmax, stabilnie
    N = x.shape[0]
    loss = -logp[np.arange(N), targets].mean()
    out = Tensor(np.array(loss, dtype=logits.data.dtype), True, (logits,), "ce")
    def _backward(g):
        p = np.exp(logp)
        p[np.arange(N), targets] -= 1.0                       # softmax - onehot
        logits._acc((p / N).reshape(logits.shape) * g)
    out._backward = _backward
    return out
```

Intuicja pochodnej: jeśli model dał prawdziwemu znakowi p=0.1, gradient logitu tego znaku = `0.1 - 1 = -0.9` (mocno "podnieś go"), a złym znakom `+p` ("obniż proporcjonalnie do tego, jak bardzo je przeceniłeś").

### Krok 2.8 — gather (pobieranie wierszy = embedding)

Model będzie brał wiersz macierzy `E` dla każdego tokena: `E[idx]`. Pochodna: gradient wraca do **tego wiersza** (a jeśli ten sam token pojawił się 5 razy w batchu — sumuje się 5 razy, stąd `np.add.at`).

```python
    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = Tensor(self.data[idx], True, (self,), "gather")
        def _backward(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)      # ten sam wiersz pobrany wiele razy -> gradienty się sumują
            self._acc(full)
        out._backward = _backward
        return out
```

Dopisz do gradchecka test na softmax, cross_entropy i gather:

```python
W = Tensor(np.random.randn(7, 5), True)
idx = np.array([1, 3, 1, 6])          # token 1 dwa razy -> test sumowania
tg = np.array([0, 4, 2, 2])
from autograd import cross_entropy
def L2(): return float(cross_entropy(W[idx].softmax() * 3.0, tg).data)
l2 = cross_entropy(W[idx].softmax() * 3.0, tg); l2.backward()
check("W gather", W, L2)
```

> ✅ **Sprawdź:** `OK`. (Wiersze 0, 2, 4, 5 macierzy `W` mają gradient 0 — nie były użyte. Sprawdź: `print(W.grad)`.)

### Krok 2.9 — optymalizator

Najprostszy: `w -= lr * grad` (SGD). Zapisz jako klasę, bo w lekcji 3 wymienisz ją na Adama:

```python
class SGD:
    def __init__(self, params, lr=0.1):
        self.params, self.lr = list(params), lr
    def zero_grad(self):
        for p in self.params:
            p.grad = None          # gradienty się AKUMULUJĄ, więc przed każdym krokiem zerujemy
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
```

## 3. Trenujemy bigram gradientem

Teraz nagroda. Ten sam bigram co w lekcji 1, ale zamiast **liczyć** tabelkę, **uczymy** się jej gradientem. Plik `moj/02_bigram_grad.py`:

```python
import numpy as np
from autograd import Tensor, cross_entropy, SGD
from data import load_dataset, get_batch

tok, train, val = load_dataset()
V = tok.vocab_size
np.random.seed(0)

W = Tensor(np.random.randn(V, V) * 0.1, requires_grad=True)   # 65x65 "logitów": wiersz = poprzedni znak
opt = SGD([W], lr=20.0)       # duży lr jest OK dla tak prostego modelu

def forward(x, y):
    logits = W[x.reshape(-1)]         # dla każdego znaku wejścia: jego wiersz W  -> (N, 65)
    return cross_entropy(logits, y)   # porównaj z prawdziwym następnym znakiem

for step in range(1001):
    x, y = get_batch(train, 32, 64)
    loss = forward(x, y)
    if step % 100 == 0:
        xv, yv = get_batch(val, 32, 256)
        print(f"step {step:3d}  train {float(loss.data):.3f}  val {float(forward(xv, yv).data):.3f}")
    opt.zero_grad()
    loss.backward()
    opt.step()
```

> ✅ **Sprawdź:** loss startuje ~4.18 (dokładnie ln 65 — zgadywanie!), po 250 krokach ~2.6, po 1000 ~2.5. Dokładnie tam, gdzie bigram z liczenia (2.48). Trwa ~1 min. Wypisz `W.data.argmax(1)` dla `q` — powinien wskazać `u`.

**To jest kluczowy moment kursu.** Tabelka z lekcji 1 i macierz wyuczona gradientem to **ten sam model**, dojście do niego innymi drogami. Gradient nie wie nic o "liczeniu par" — tylko minimalizuje zaskoczenie. A mimo to odkrywa tę samą statystykę. Tę samą metodę zastosujesz do modelu, dla którego nie ma tabelki.

## 4. Ćwiczenia

1. **(obowiązkowe)** Zmień `lr` na 0.1, 1.0, 20, 200. Co się dzieje z krzywą? Zapisz w notatkach: za mały lr → ?, za duży → ?
2. Zaimplementuj `gelu` (wzór w `rozwiazania/autograd.py`) i dodaj go do gradchecka. GELU to nieliniowość używana w GPT-2.
3. Zaimplementuj `layernorm(gamma, beta)`: `(x - mean) / sqrt(var + eps) * gamma + beta` po ostatniej osi. Backward jest najtrudniejszy w całym silniku — możesz przepisać z rozwiązań, **ale dodaj do gradchecka** i przekonaj się, że działa. Będzie potrzebny w lekcji 5.
4. Zepsuj celowo backward w `__mul__` (np. zamień `other.data` na `self.data`). Uruchom gradcheck. Zobacz, jak test to wyłapuje. Napraw.
5. Dlaczego w `SGD.zero_grad` ustawiamy `None`, a nie `0`? (Podpowiedź: `_acc`.)

## 5. Najczęstsze błędy

- **Gradient 2× za duży / za mały** → gdzieś `_acc` przypisuje zamiast dodawać, albo brak `.copy()` i dwa tensory dzielą jedną tablicę.
- **`ValueError: operands could not be broadcast`** w backward → brakuje `_unbroadcast`.
- **Loss rośnie zamiast spadać** → znak: `p.data -= lr*grad` (minus!), albo lr za duży.
- **Gradcheck OK dla float64, ale trening dziwny** → w treningu używasz float32 (celowo, szybciej); to normalne, że gradcheck w float32 daje błąd ~1e-3. Testuj w float64.
- **Wolno** → sprawdź, czy przypadkiem nie masz pętli `for` po elementach tablicy. W backward mają być tylko operacje NumPy na całych tablicach.

## Co zapamiętać

- Trening = policz loss → policz gradient każdej wagi → przesuń wagi pod górkę w dół → powtórz.
- Autograd = każda operacja zapamiętuje dzieci i umie rozdać gradient (reguła łańcuchowa). `backward()` idzie po grafie od loss do wag.
- Pochodna numeryczna jest za wolna do treningu, ale to jedyny sposób, żeby **wiedzieć**, że backward jest poprawny.
- Gradient odkrywa tę samą statystykę co liczenie — ale działa też tam, gdzie liczyć się nie da.

➡️ [Lekcja 3 — Embedding i sieć z kontekstem (MLP)](03_mlp.md)
