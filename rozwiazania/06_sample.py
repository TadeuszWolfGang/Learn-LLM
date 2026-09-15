"""
LEKCJA 6 - generowanie tekstu z checkpointa; temperatura i top-k.

    python3 06_sample.py out/best.npz
    python3 06_sample.py out/best.npz --temp 0.3          # zachowawczo (powtarza się)
    python3 06_sample.py out/best.npz --temp 1.5          # chaos
    python3 06_sample.py out/best.npz --top_k 5 --prompt "ROMEO:"
"""
import argparse
import numpy as np
from data import load_dataset
from model import GPT

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--data", default=None)
p.add_argument("--prompt", default="\n")
p.add_argument("--n", type=int, default=400)
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=None)
p.add_argument("--seed", type=int, default=None)
a = p.parse_args()
if a.seed is not None:
    np.random.seed(a.seed)

tok, _, _ = load_dataset() if a.data is None else load_dataset(a.data)
m = GPT.load(a.ckpt)
print(f"model: {m.n_params():,} parametrow, T={a.temp}, top_k={a.top_k}\n" + "-" * 60)
print(tok.decode(m.generate(tok.encode(a.prompt), a.n, a.temp, a.top_k)))
