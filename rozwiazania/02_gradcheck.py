"""
LEKCJA 2 - test: czy nasz backward liczy to samo, co pochodna numeryczna?

Pochodna numeryczna: dL/dw ≈ (L(w+h) - L(w-h)) / 2h
Jeśli autograd się zgadza z tym do ~1e-5, to backward jest poprawny.
To najważniejszy test w całym repo: jeśli tu jest błąd, cały trening jest bez sensu.
"""
import numpy as np
from autograd import Tensor, cross_entropy
from model import GPT, GPTConfig

np.random.seed(0)


def numeric_grad(f, x, h=1e-5):
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
    print(f"  {name:22s} max rel err = {err:.2e}  {'OK' if err < tol else 'BLAD!'}")
    return err < tol


# ---- 1. pojedyncze operacje (float64, żeby błąd numeryczny był mały) ----
print("Operacje elementarne:")
ok = True
a = Tensor(np.random.randn(3, 4), True)
b = Tensor(np.random.randn(4, 5), True)
g = Tensor(np.random.randn(4), True)
bt = Tensor(np.random.randn(4), True)


def loss_fn():
    y = ((a @ b).tanh() * 2 + 1).softmax(-1)
    z = a.layernorm(g, bt).gelu() @ b
    return float((y ** 2).sum().data + z.relu().sum().data + (a.exp() * 0.1).sum().data)


L = ((a @ b).tanh() * 2 + 1).softmax(-1) ** 2
L = L.sum() + (a.layernorm(g, bt).gelu() @ b).relu().sum() + (a.exp() * 0.1).sum()
L.backward()
for n, t in [("a (matmul/ln/exp)", a), ("b", b), ("gamma", g), ("beta", bt)]:
    ok &= check(n, t, loss_fn)

# ---- 2. cały GPT: cross-entropy po parametrach ----
print("Caly mikro-GPT (float64):")
cfg = GPTConfig(vocab_size=7, block_size=5, n_layer=1, n_head=2, n_embd=8)
m = GPT(cfg)
for p in m.params():
    p.data = p.data.astype(np.float64)
    if p.data.ndim >= 2:
        p.data = np.random.randn(*p.data.shape) * 0.5   # większe wagi = ciekawszy test
for blk in m.blocks:
    blk.attn.mask = blk.attn.mask.astype(np.float64)
x = np.random.randint(0, 7, size=(2, 5))
y = np.random.randint(0, 7, size=(2, 5))


def gpt_loss():
    return float(m(x, y)[1].data)


_, loss = m(x, y)
loss.backward()
names = ["tok_emb", "pos_emb"] + [f"blk_p{i}" for i in range(len(m.blocks[0].params()))] + ["ln_f.g", "ln_f.b", "head.w"]
for n, p in zip(names, m.params()):
    ok &= check(n, p, gpt_loss)

print("\nWSZYSTKO OK" if ok else "\nCOS NIE GRA - sprawdz backward w autograd.py")
