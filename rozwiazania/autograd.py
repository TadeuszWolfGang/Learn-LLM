"""
Mini silnik autograd na tablicach NumPy.  (~250 linii)

To jest "micrograd, ale na tensorach". Każdy Tensor pamięta:
  - data      : tablica NumPy z wartościami
  - grad      : tablica NumPy z gradientem dL/d(data), wypełniana w backward()
  - _backward : funkcja, która rozdaje gradient "w dół" do dzieci
  - _prev     : tensory, z których ten powstał (graf obliczeń)

Dlaczego nie micrograd.py Karpathy'ego?  Micrograd liczy na pojedynczych
liczbach (skalarach).  Transformer o 1M parametrów w micrograd = miliony
obiektów Pythona na jeden krok -> godziny na jeden batch.  Tutaj jeden
Tensor = cała macierz, a mnożenie robi NumPy (C/BLAS), więc trening
mikro-GPT trwa minuty.

Konwencja: wszystko czego uczysz się w lekcji 2 jest w tym pliku.
Nie ma tu magii: przeczytaj forward i backward każdej operacji.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Suma gradientu po osiach, które NumPy 'rozciągnął' (broadcasting).
    Np. bias (C,) dodany do (B,T,C): gradient biasu = suma po B i T."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda g: None
        self._prev = tuple(_children)
        self._op = _op

    # ---------- pomocnicze ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def _acc(self, g):
        """Dodaj gradient (może przyjść z kilku ścieżek -> sumujemy)."""
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype)   # kopia: g może być cudzym buforem
        else:
            self.grad += g

    @staticmethod
    def _wrap(x):
        if isinstance(x, Tensor):
            return x
        if isinstance(x, (int, float, np.number)):
            # skalar zawsze jako float32: np.float64 (np. 1/np.sqrt(d)) awansowałby cały model do float64
            return Tensor(np.asarray(x, dtype=np.float32))
        return Tensor(x)

    # ---------- operacje elementarne ----------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, True, (self, other), "+")

        def _backward(g):
            self._acc(_unbroadcast(g, self.shape))
            other._acc(_unbroadcast(g, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, True, (self, other), "*")

        def _backward(g):
            self._acc(_unbroadcast(g * other.data, self.shape))
            other._acc(_unbroadcast(g * self.data, other.shape))
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        return self * (other ** -1)

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "potęga tylko przez liczbę"
        out = Tensor(self.data ** p, True, (self,), f"**{p}")

        def _backward(g):
            if p == 2:
                self._acc(g * 2 * self.data)
            elif p == -1:
                self._acc(-g / (self.data * self.data))
            else:
                self._acc(g * p * self.data ** (p - 1))
        out._backward = _backward
        return out

    def exp(self):
        y = np.exp(self.data)
        out = Tensor(y, True, (self,), "exp")

        def _backward(g):
            self._acc(g * y)          # y, nie out.data: closure nie może trzymać out (cykl)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), True, (self,), "log")

        def _backward(g):
            self._acc(g / self.data)
        out._backward = _backward
        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Tensor(y, True, (self,), "tanh")

        def _backward(g):
            self._acc(g * (1 - y * y))
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), True, (self,), "relu")

        def _backward(g):
            self._acc(g * (self.data > 0))
        out._backward = _backward
        return out

    def gelu(self):
        """GELU (wersja tanh, jak w GPT-2)."""
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        x2 = x * x                                   # x*x zamiast x**2: 10x szybciej w NumPy
        t = np.tanh(c * (x + 0.044715 * x2 * x))
        out = Tensor(0.5 * x * (1 + t), True, (self,), "gelu")

        def _backward(g):
            dinner = c * (1 + 3 * 0.044715 * x2)
            dx = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
            self._acc(g * dx)
        out._backward = _backward
        return out

    # ---------- redukcje i kształty ----------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), True, (self,), "sum")

        def _backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), True, (self,), "reshape")

        def _backward(g):
            self._acc(g.reshape(self.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes):
        """Permutacja osi, np. x.transpose(0, 2, 1, 3)."""
        axes = tuple(axes)
        out = Tensor(self.data.transpose(axes), True, (self,), "T")
        inv = np.argsort(axes)

        def _backward(g):
            self._acc(g.transpose(inv))
        out._backward = _backward
        return out

    def swap_last2(self):
        """Zamiana dwóch ostatnich osi (do K^T w attention)."""
        nd = self.data.ndim
        axes = list(range(nd))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    # ---------- algebra liniowa ----------
    def __matmul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, True, (self, other), "@")

        def _backward(g):
            # C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            a, b = self.data, other.data
            bt = np.swapaxes(b, -1, -2) if b.ndim > 1 else b
            at = np.swapaxes(a, -1, -2) if a.ndim > 1 else a
            self._acc(_unbroadcast(g @ bt, self.shape))
            other._acc(_unbroadcast(at @ g, other.shape))
        out._backward = _backward
        return out

    # ---------- operacje "sklejone" (szybsze i stabilniejsze) ----------
    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)  # stabilność numeryczna
        e = np.exp(x)
        y = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(y, True, (self,), "softmax")

        def _backward(g):
            # dx = y * (dy - sum(dy*y))
            self._acc(y * (g - (g * y).sum(axis=axis, keepdims=True)))
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        """LayerNorm po ostatniej osi.  gamma, beta: Tensor (C,)."""
        x = self.data
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * rstd
        out = Tensor(xhat * gamma.data + beta.data, True, (self, gamma, beta), "ln")

        def _backward(g):
            C = x.shape[-1]
            gamma._acc(_unbroadcast(g * xhat, gamma.shape))
            beta._acc(_unbroadcast(g, beta.shape))
            dxhat = g * gamma.data
            dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                         - xhat * (dxhat * xhat).mean(-1, keepdims=True))
            self._acc(dx)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Gather: emb[idx] (embedding lookup).  idx = tablica intów."""
        idx = np.asarray(idx)
        out = Tensor(self.data[idx], True, (self,), "gather")

        def _backward(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)   # ten sam wiersz może być pobrany wiele razy -> sumuj
            self._acc(full)
        out._backward = _backward
        return out

    def dropout(self, p, training=True):
        if not training or p == 0.0:
            return self
        mask = (np.random.rand(*self.shape) >= p) / (1.0 - p)
        return self * mask.astype(self.dtype)

    # ---------- backward ----------
    def backward(self):
        """Odwrotna kolejność topologiczna: od loss do liści.
        Bez rekurencji: (1) głęboki graf nie przekroczy limitu rekurencji Pythona,
        (2) rekurencyjna funkcja wewnętrzna tworzyłaby cykl referencji, który trzyma
            CAŁY graf w pamięci aż do uruchomienia GC -> wyciek setek MB na krok."""
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            t, done = stack.pop()
            if done:
                topo.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for c in t._prev:
                if id(c) not in seen:
                    stack.append((c, False))
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward(t.grad)


# ---------- funkcje straty ----------
def cross_entropy(logits, targets):
    """logits: Tensor (N, V); targets: tablica intów (N,).  Zwraca średnią stratę.
    Sklejone softmax + log + NLL: stabilnie numerycznie i jeden backward."""
    targets = np.asarray(targets).reshape(-1)
    x = logits.data.reshape(-1, logits.shape[-1])
    x = x - x.max(-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(-1, keepdims=True))
    N = x.shape[0]
    loss = -logp[np.arange(N), targets].mean()
    out = Tensor(np.array(loss, dtype=logits.dtype), True, (logits,), "ce")

    def _backward(g):
        p = np.exp(logp)
        p[np.arange(N), targets] -= 1.0        # softmax - onehot
        logits._acc((p / N).reshape(logits.shape) * g)
    out._backward = _backward
    return out


# ---------- optymalizatory ----------
class SGD:
    def __init__(self, params, lr=0.1):
        self.params, self.lr = list(params), lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad


class AdamW:
    """AdamW: Adam + weight decay 'odklejony' od gradientu (jak w GPT-2/nanoGPT)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
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
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            if self.wd and p.data.ndim >= 2:       # decay tylko macierzy, nie biasów/LN
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_grad_norm(params, max_norm=1.0):
    """Obcięcie normy gradientu (stabilizuje trening transformera)."""
    total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in params if p.grad is not None))
    if total > max_norm:
        s = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= s
    return total
