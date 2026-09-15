"""
LEKCJA 0 - sprawdzenie środowiska na iPadzie.

    python3 00_check_env.py

Sprawdza NumPy/matplotlib, pobiera Tiny Shakespeare, mierzy prędkość
Twojego CPU i szacuje, ile potrwa trening mikro-GPT z lekcji 5.
"""
import sys, time, platform, os
print("Python   :", sys.version.split()[0], "|", platform.platform())
try:
    import numpy as np
    print("NumPy    :", np.__version__, "OK")
except ImportError:
    sys.exit("BRAK NumPy. a-Shell: zainstaluj pełną wersję (nie 'mini'). Pythonista/Carnets mają NumPy wbudowane.")
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__, "OK")
except ImportError:
    print("matplotlib: BRAK (wykresy nie zadziałają, trening tak)")

from data import download_if_missing, load_dataset
path = download_if_missing()
tok, tr, va = load_dataset(path)
print(f"Dane     : {len(tr)+len(va):,} znaków, vocab={tok.vocab_size}  ({path})")

# benchmark: mnożenie macierzy (to robi 90% pracy w transformerze)
a = np.random.randn(1024, 1024).astype(np.float32)
a @ a
t = time.time(); n = 5
for _ in range(n):
    a @ a
gflops = n * 2 * 1024 ** 3 / (time.time() - t) / 1e9
print(f"matmul   : {gflops:.1f} GFLOP/s (float32, 1024x1024)")

# benchmark: jeden krok treningowy domyślnego modelu z lekcji 5
from model import GPT, GPTConfig
from autograd import AdamW, clip_grad_norm
from data import get_batch
m = GPT(GPTConfig(tok.vocab_size, 64, 2, 4, 64)); ps = m.params(); opt = AdamW(ps, 3e-3)
def step():
    x, y = get_batch(tr, 64, 32); _, l = m(x, y, True)
    opt.zero_grad(); l.backward(); clip_grad_norm(ps); opt.step()
step(); t = time.time()
for _ in range(5):
    step()
s = (time.time() - t) / 5
print(f"krok GPT : {s*1000:.0f} ms  (model {m.n_params():,} parametrów, batch 32x64)")
print(f"Szacunek : lekcja 5 (2000 kroków) ≈ {2000*s/60:.1f} min;  większy model (4x128, 3000 kroków) ≈ {3000*s*4/60:.0f} min")
print("\nJeśli wszystko wyżej jest OK -> możesz zaczynać lekcję 1.")
