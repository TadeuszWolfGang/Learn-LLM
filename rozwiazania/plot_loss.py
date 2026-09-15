"""
Wykres train vs val loss z log.csv.  Zapisuje PNG obok logu.
    python3 plot_loss.py out/log.csv            # -> out/loss.png
    python3 plot_loss.py out_a/log.csv out_b/log.csv   # porównanie kilku treningów

a-Shell:   open out/loss.png   (podgląd Quick Look)
Pythonista/Carnets/Juno: wykres pokaże się w konsoli/notebooku.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = sys.argv[1:] or ["out/log.csv"]
plt.figure(figsize=(7, 4))
for path in paths:
    d = np.genfromtxt(path, delimiter=",", names=True)
    lab = os.path.basename(os.path.dirname(path)) or path
    plt.plot(d["step"], d["train"], label=f"{lab} train")
    plt.plot(d["step"], d["val"], "--", label=f"{lab} val")
plt.axhline(np.log(65), color="gray", lw=0.8, ls=":", label="ln(65)=4.17 (zgadywanie)")
plt.xlabel("krok"); plt.ylabel("cross-entropy (nats/znak)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
out = os.path.join(os.path.dirname(paths[0]) or ".", "loss.png")
plt.savefig(out, dpi=130)
print("zapisano", out)
try:
    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass
