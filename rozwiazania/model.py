"""
Mikro-GPT w czystym NumPy (na silniku z autograd.py).

Architektura (dokładnie jak GPT-2, tylko malutka):

  tokens ──> tok_emb + pos_emb ──> [Block] x n_layer ──> LayerNorm ──> Linear(vocab) ──> logits
                                      │
            Block:  x = x + Attention(LayerNorm(x))     <- "pre-LN", residual
                    x = x + FFN(LayerNorm(x))

Wszystko co "widzisz na własne oczy" w lekcjach 4-6 jest tutaj.
"""
import numpy as np
from autograd import Tensor, cross_entropy


class Linear:
    def __init__(self, n_in, n_out, std=0.02, bias=True):
        self.w = Tensor(np.random.randn(n_in, n_out).astype(np.float32) * std, requires_grad=True)
        self.b = Tensor(np.zeros(n_out, dtype=np.float32), requires_grad=True) if bias else None

    def __call__(self, x):
        y = x @ self.w
        return y + self.b if self.b is not None else y

    def params(self):
        return [self.w] + ([self.b] if self.b is not None else [])


class LayerNorm:
    def __init__(self, n):
        self.g = Tensor(np.ones(n, dtype=np.float32), requires_grad=True)
        self.b = Tensor(np.zeros(n, dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        return x.layernorm(self.g, self.b)

    def params(self):
        return [self.g, self.b]


class CausalSelfAttention:
    """Multi-head causal self-attention.  To jest serce transformera."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        assert n_embd % n_head == 0
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv = Linear(n_embd, 3 * n_embd)          # jedna macierz dla Q, K, V
        self.proj = Linear(n_embd, n_embd)
        self.dropout = dropout
        # maska przyczynowa: token t widzi tylko tokeny <= t.  -1e9 => po softmaxie 0.
        tril = np.tril(np.ones((block_size, block_size), dtype=np.float32))
        self.mask = np.where(tril == 1, 0.0, -1e9).astype(np.float32)
        self.last_att = None                           # do podglądu w lekcji 4/6

    def __call__(self, x, training=False):
        B, T, C = x.shape
        qkv = self.qkv(x)                                        # (B, T, 3C)
        q = qkv.reshape(B, T, 3, self.n_head, self.hs)
        q = q.transpose(2, 0, 3, 1, 4)                           # (3, B, nh, T, hs)
        # "wyciągamy" Q, K, V jako trzy widoki jednego tensora
        q, k, v = _split3(q)
        att = (q @ k.swap_last2()) * (1.0 / np.sqrt(self.hs))    # (B, nh, T, T)
        att = att + self.mask[:T, :T]                            # zasłoń przyszłość
        att = att.softmax(-1)                                    # wagi uwagi sumują się do 1
        self.last_att = att.data
        att = att.dropout(self.dropout, training)
        y = att @ v                                              # (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)             # sklej głowy
        return self.proj(y).dropout(self.dropout, training)

    def params(self):
        return self.qkv.params() + self.proj.params()


def _split3(t):
    """Rozbija tensor (3, ...) na trzy tensory (…) z gradientem."""
    outs = []
    for i in range(3):
        o = Tensor(t.data[i], True, (t,), f"sel{i}")

        def _backward(g, i=i):
            gg = np.zeros_like(t.data)
            gg[i] = g
            t._acc(gg)
        o._backward = _backward
        outs.append(o)
    return outs


class MLP:
    """Feed-forward: rozszerz 4x, nieliniowość, zwęź.  Tu 'siedzi' większość parametrów."""

    def __init__(self, n_embd, dropout=0.0):
        self.fc = Linear(n_embd, 4 * n_embd)
        self.proj = Linear(4 * n_embd, n_embd)
        self.dropout = dropout

    def __call__(self, x, training=False):
        return self.proj(self.fc(x).gelu()).dropout(self.dropout, training)

    def params(self):
        return self.fc.params() + self.proj.params()


class Block:
    def __init__(self, cfg):
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)
        self.cfg = cfg

    def __call__(self, x, training=False):
        # residual: x + f(x).  Lekcja 6: usuń "x +" i zobacz, co się dzieje z treningiem.
        if self.cfg.use_residual:
            x = x + self.attn(self.ln1(x), training)
            x = x + self.mlp(self.ln2(x), training)
        else:
            x = self.attn(self.ln1(x), training)
            x = self.mlp(self.ln2(x), training)
        return x

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()


class GPTConfig:
    def __init__(self, vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64,
                 dropout=0.0, use_residual=True, use_pos_emb=True):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.dropout, self.use_residual, self.use_pos_emb = dropout, use_residual, use_pos_emb

    def to_dict(self):
        return dict(self.__dict__)


class GPT:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tok_emb = Tensor(np.random.randn(cfg.vocab_size, cfg.n_embd).astype(np.float32) * 0.02,
                              requires_grad=True)
        self.pos_emb = Tensor(np.random.randn(cfg.block_size, cfg.n_embd).astype(np.float32) * 0.02,
                              requires_grad=True)
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
        """idx: (B, T) inty.  Zwraca (logits Tensor (B,T,V), loss Tensor lub None)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.tok_emb[idx]                                    # (B, T, C)
        if self.cfg.use_pos_emb:
            x = x + self.pos_emb[np.arange(T)]                   # (T, C) broadcast
        x = x.dropout(self.cfg.dropout, training)
        for b in self.blocks:
            x = b(x, training)
        x = self.ln_f(x)
        logits = self.head(x)                                    # (B, T, V)
        loss = None
        if targets is not None:
            loss = cross_entropy(logits.reshape(B * T, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=np.random):
        """Autoregresja: przewidź następny token, doklej, powtórz."""
        idx = np.array(idx, dtype=np.int64).reshape(1, -1)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits.data[0, -1] / max(temperature, 1e-6)
            if top_k is not None:
                kth = np.sort(logits)[-top_k]
                logits = np.where(logits < kth, -np.inf, logits)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = rng.choice(len(p), p=p)
            idx = np.concatenate([idx, [[nxt]]], axis=1)
        return idx[0]

    # ---------- checkpointy ----------
    def save(self, path):
        arrs = {f"p{i}": p.data for i, p in enumerate(self.params())}
        np.savez(path, cfg=np.array([repr(self.cfg.to_dict())]), **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = GPTConfig(**eval(str(z["cfg"][0])))
        m = cls(cfg)
        for i, p in enumerate(m.params()):
            p.data[...] = z[f"p{i}"]
        return m
