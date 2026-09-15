"""
LEKCJA 6/7 - co "widzą" głowy uwagi wytrenowanego mikro-GPT?

    python3 07_visualize_attention.py out/best.npz
    python3 07_visualize_attention.py out/best.npz --text "ROMEO:\nBut soft, what light"

Rysuje mapę uwagi każdej głowy w każdej warstwie dla podanego tekstu.
Typowe odkrycia: głowa patrząca na poprzedni znak, głowa patrząca na ostatnią
spację (początek słowa), głowa patrząca na początek linii.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_dataset
from model import GPT

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--data", default=None)
p.add_argument("--text", default="First Citizen:\nBefore we proceed any")
p.add_argument("--out", default="attention_heads.png")
a = p.parse_args()

tok, _, _ = load_dataset() if a.data is None else load_dataset(a.data)
m = GPT.load(a.ckpt)
text = a.text.encode().decode("unicode_escape")[: m.cfg.block_size]
ids = tok.encode(text).reshape(1, -1)
m(ids)
labels = [c if c != "\n" else "⏎" for c in text]
L, H = m.cfg.n_layer, m.cfg.n_head
fig, axes = plt.subplots(L, H, figsize=(3.2 * H, 3.2 * L), squeeze=False)
for l, blk in enumerate(m.blocks):
    att = blk.attn.last_att[0]                       # (nh, T, T)
    for h in range(H):
        ax = axes[l][h]
        ax.imshow(att[h], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"warstwa {l} głowa {h}", fontsize=9)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
plt.tight_layout(); plt.savefig(a.out, dpi=120)
print("zapisano", a.out)

# statystyka: na jaką odległość wstecz średnio patrzy każda głowa?
T = len(text)
dist = np.arange(T)[:, None] - np.arange(T)[None, :]
for l, blk in enumerate(m.blocks):
    for h in range(H):
        A = blk.attn.last_att[0, h]
        print(f"warstwa {l} głowa {h}: średnia odległość wstecz = {(A * dist).sum(1).mean():.2f} znaków")
