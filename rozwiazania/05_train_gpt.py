"""
LEKCJA 5 - trening mikro-GPT na Tiny Shakespeare.

Uruchom:
    python3 05_train_gpt.py                       # domyślny model ~200k parametrów, ~2-4 min
    python3 05_train_gpt.py --n_layer 4 --n_embd 128 --steps 3000   # większy (~800k), ~15-30 min
    python3 05_train_gpt.py --resume out/ckpt.npz --steps 1000      # dotrenuj z checkpointa

Co oglądać: kolumny "train" i "val" w logu.  Jak val przestaje spadać
a train dalej spada -> overfitting.  Wykres: python3 plot_loss.py out/log.csv
"""
import argparse, os, time
import numpy as np
from data import load_dataset, get_batch, download_if_missing
from model import GPT, GPTConfig
from autograd import AdamW, clip_grad_norm

p = argparse.ArgumentParser()
p.add_argument("--data", default=None, help="plik tekstowy (domyślnie Tiny Shakespeare)")
p.add_argument("--n_layer", type=int, default=2)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_embd", type=int, default=64)
p.add_argument("--block_size", type=int, default=64)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--min_lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--eval_batches", type=int, default=10)
p.add_argument("--no_residual", action="store_true", help="eksperyment: wyłącz połączenia residualne")
p.add_argument("--no_pos_emb", action="store_true", help="eksperyment: wyłącz pozycję")
p.add_argument("--train_frac", type=float, default=1.0,
               help="użyj tylko ułamka danych treningowych (np. 0.01 = overfitting na żądanie)")
p.add_argument("--out", default="out")
p.add_argument("--resume", default=None)
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
if args.data is None:
    download_if_missing()
    tok, train_ids, val_ids = load_dataset()
else:
    tok, train_ids, val_ids = load_dataset(args.data)
if args.train_frac < 1.0:
    train_ids = train_ids[: max(args.block_size + 2, int(len(train_ids) * args.train_frac))]
print(f"vocab={tok.vocab_size}  train={len(train_ids):,} znakow  val={len(val_ids):,} znakow")

if args.resume:
    model = GPT.load(args.resume)
    print("wczytano", args.resume)
else:
    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd,
                    args.dropout, use_residual=not args.no_residual, use_pos_emb=not args.no_pos_emb)
    model = GPT(cfg)
params = model.params()
print(f"parametry: {model.n_params():,}   (~{model.n_params()*4*4/1e6:.1f} MB z Adamem)")
opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def lr_at(step):
    """warmup liniowy, potem cosinus do min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * t))


def estimate_loss(ids, n):
    ls = []
    for _ in range(n):
        x, y = get_batch(ids, model.cfg.block_size, args.batch_size)
        ls.append(float(model(x, y)[1].data))
    return float(np.mean(ls))


log_path = os.path.join(args.out, "log.csv")
with open(log_path, "a" if args.resume else "w") as f:
    if not args.resume:
        f.write("step,train,val,lr,sec\n")

t0 = time.time()
best_val = float("inf")
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        tr, va = estimate_loss(train_ids, args.eval_batches), estimate_loss(val_ids, args.eval_batches)
        el = time.time() - t0
        print(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | lr {lr_at(step):.1e} | {el:6.0f}s")
        with open(log_path, "a") as f:
            f.write(f"{step},{tr:.4f},{va:.4f},{lr_at(step):.2e},{el:.0f}\n")
        model.save(os.path.join(args.out, "ckpt.npz"))
        if va < best_val:
            best_val = va
            model.save(os.path.join(args.out, "best.npz"))
        if step == args.steps:
            break
    x, y = get_batch(train_ids, model.cfg.block_size, args.batch_size)
    _, loss = model(x, y, training=True)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm(params, 1.0)
    opt.lr = lr_at(step)
    opt.step()

print(f"\nkoniec. najlepszy val = {best_val:.3f}.  checkpoint: {args.out}/best.npz")
print("probka tekstu:")
ids = model.generate(tok.encode("\n"), 300, temperature=0.8)
print(tok.decode(ids))
