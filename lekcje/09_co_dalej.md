# Lekcja 9 — Most dalej: BPE, fine-tuning, LoRA, co na Maca

**Czas:** ~1 h (czytanie + rachunki) · **Cel:** rozumiesz, czym Twój mikro-GPT różni się od Qwen/Llama (i czym nie), co to jest fine-tuning i LoRA **na poziomie macierzy, które sam napisałeś**, umiesz policzyć, ile pamięci potrzebuje LoRA 3B/8B, i masz plan na następny krok poza iPadem.

---

## 1. Twój GPT vs prawdziwy: lista różnic

| Element | Twój mikro-GPT | Qwen2.5 / Llama 3 | Czy zmienia zasadę? |
|---------|----------------|-------------------|---------------------|
| Tokenizer | 65 znaków | BPE, ~128k–150k podsłów | nie — tylko `vocab_size` |
| Pozycja | wyuczony `pos_emb` | RoPE (obrót Q i K o kąt zależny od pozycji) | nie — inna metoda, ten sam cel |
| Normalizacja | LayerNorm | RMSNorm (bez odejmowania średniej) | nie |
| Nieliniowość | GELU | SwiGLU (bramkowana; 3 macierze zamiast 2) | nie |
| Attention | MHA, 4 głowy | GQA (kilka głów Q dzieli jedną K,V — oszczędza KV-cache) | nie |
| Rozmiar | 2 warstwy × 64 | 28–32 warstw × 3072–4096 | tylko skala |
| Dane | 1 MB | 15–18 bilionów tokenów | tylko skala |
| Trening | AdamW, warmup, cosinus, clipping | **to samo** + równoległość na tysiącach GPU | nie |
| Po pretrainingu | koniec | SFT → RLHF/DPO (instruct) | nowy etap, ta sama pętla treningowa |

Każdy z tych elementów to wariant czegoś, co napisałeś. Gdy przeczytasz "RoPE" albo "GQA" w opisie modelu, wiesz, **gdzie w kodzie** to siedzi.

### BPE w 5 minut

Char-level: `"retransmisje"` = 12 tokenów. Model musi na każdym z 12 kroków "pamiętać", że jest w środku tego słowa. BPE (byte-pair encoding) buduje słownik iteracyjnie: zacznij od bajtów, znajdź najczęstszą parę sąsiadów, sklej w nowy token, powtórz 100k razy. Efekt: `"retransmisje"` → `["retrans", "mis", "je"]`, 3 tokeny. Ten sam tekst = 4× mniej kroków, kontekst 4× "dłuższy" w znakach za tę samą cenę.

Ćwiczenie w głowie: dlaczego liczby są problemem dla BPE? (`"310"` może być jednym tokenem, `"3100"` dwoma `["310","0"]` — model widzi je jako niepowiązane symbole. Stąd słabość LLM w arytmetyce.)

## 2. Fine-tuning na poziomie macierzy

**Pretraining**: pętla z lekcji 5 na 15 bilionach tokenów internetu. Kosztuje miliony dolarów. Nigdy tego nie robisz.

**Fine-tuning (SFT)**: **ta sama pętla**, te same wagi na starcie (z pretrainingu), ale dane to Twoje pary "prompt → odpowiedź", `lr` 10–100× mniejszy, kilkaset–kilka tysięcy kroków. Loss liczysz zwykle tylko na tokenach odpowiedzi (maska na `targets`: -1 tam, gdzie prompt — mała zmiana w `cross_entropy`).

Problem: pełny fine-tuning 3B = trzeba trzymać w pamięci wagi + gradient + m + v Adama = 16 bajtów/parametr = **48 GB**. 8B = 128 GB. Nie na iPadzie, nie na Macu 16 GB.

### LoRA: dlaczego działa i ile kosztuje

Pomysł: nie zmieniaj macierzy `W` (np. `qkv.w` rozmiaru 4096×12288). Zamiast tego dodaj obok **dwie małe**: `A` (4096×r) i `B` (r×12288), z `r`=8–64, i licz `y = x @ W + x @ A @ B`. Trenujesz tylko `A`, `B`. `W` zamrożone (bez gradientu, bez `m`, `v`).

W Twoim kodzie to dosłownie:

```python
class LoRALinear:
    def __init__(self, base: Linear, r=8, alpha=16):
        self.base = base                       # W zamrożone: nie trafia do params()
        n_in, n_out = base.w.shape
        self.A = Tensor(np.random.randn(n_in, r).astype(np.float32) / np.sqrt(n_in), True)
        self.B = Tensor(np.zeros((r, n_out), dtype=np.float32), True)   # start od 0 -> model na starcie = bazowy
        self.scale = alpha / r
    def __call__(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale
    def params(self):
        return [self.A, self.B]                # tylko to trenujemy
```

Parametry do trenowania: `r·(n_in + n_out)` zamiast `n_in·n_out`. Dla 4096×12288 i r=16: 262k zamiast 50M — **200× mniej**. Pamięć: wagi bazowe (mogą być 4-bit! → QLoRA) + malutki Adam.

**Ćwiczenie (obowiązkowe, na papierze):** Qwen2.5-3B, 36 warstw, `n_embd`=2048, LoRA r=16 na `q,k,v,o` (4 macierze ~2048×2048 na warstwę) + `gate,up,down` (2048×11008 ×3). Policz liczbę parametrów LoRA i pamięć na Adam (16 B/param). Potem wagi bazowe: 3B × 0.5 B (4-bit) ≈ 1.5 GB, albo × 2 B (fp16) = 6 GB. Plus aktywacje na batch (zależy od długości sekwencji, przy 2k tokenów ~2–4 GB). Suma → co się mieści w 16 GB? A 8B?

Odpowiedź, którą powinieneś dostać: LoRA 3B w fp16 mieści się w ~10–12 GB, 8B w 4-bit (QLoRA) w ~10–14 GB. **Chip iPada by to udźwignął.** Nie udźwignie tego iPadOS: brak PyTorch/MLX-Python, brak procesów w tle na 2–6 h, limit pamięci na aplikację. Dlatego Mac.

### Ćwiczenie praktyczne na iPadzie (opcjonalne, 1 h)

Zaimplementuj `LoRALinear` w `moj/model.py`, weź wytrenowany `out/best.npz` z lekcji 5, zamroź wszystko, podepnij LoRA r=4 pod `qkv` i `proj` w każdym bloku, i dotrenuj **tylko LoRA** 300 kroków na `data/incidents.txt`. Obserwuj:
- ile parametrów trenujesz (powinno być ~3–5% modelu),
- val loss na incydentach spada (model "przestawia się" na domenę),
- wygeneruj z promptem `ROMEO:` — Szekspir nadal "siedzi" w zamrożonych wagach, choć osłabiony.

To jest cały LoRA. Na Macu robi to za Ciebie `mlx_lm.lora`, ale wiesz, co robi.

## 3. Plan na Maca (albo DGX Spark / wynajęte GPU)

Kolejność, gdy arkusz z lekcji 8 dał 20–31/40:

1. **Dane**: 200–500 przykładów w formacie z `prompty/rca_fewshot.md`, JSONL `{"prompt": ..., "completion": ...}`. 10% odłóż na val (lekcja 1!). Zbalansuj kategorie.
2. **Narzędzie**: na Macu `pip install mlx-lm`; `mlx_lm.lora --model Qwen/Qwen2.5-3B-Instruct --train --data ./dane --iters 600 --batch-size 4 --lora-layers 16`. Na Linux/GPU: Unsloth albo HF `peft` + `trl`.
3. **Oglądaj to samo, co w lekcji 5**: train vs val loss. Rozjazd = overfitting na 300 przykładach (bardzo częste; lek: mniej iteracji, mniejszy r, więcej danych).
4. **Ewaluacja**: ten sam arkusz z lekcji 8 na tych samych 10 testowych. Porównaj z few-shot. Jeśli LoRA nie bije few-shot o ≥ 5 pkt — nie warto jej utrzymywać.
5. **Wdrożenie**: `mlx_lm.fuse` scala LoRA z wagami → GGUF → wraca na iPada do PocketPal. Koło się zamyka: model fine-tunowany na Macu, uruchamiany na iPadzie.

Czasy na Macu M-series 16 GB: 3B LoRA, 500 przykładów, 600 iteracji ≈ 20–40 min. 8B QLoRA ≈ 1–2 h.

## 4. Co czytać dalej (w tej kolejności)

1. Karpathy, *Let's build GPT: from scratch, in code* (YouTube, 2 h) — to samo, co zrobiłeś, w PyTorchu. Będziesz rozumiał każdą linijkę.
2. Karpathy, *nanoGPT* (GitHub) — Twój `model.py` to jego uproszczona kopia. Porównaj.
3. *The Illustrated Transformer* (Jay Alammar) — obrazki do tego, co policzyłeś w lekcji 4.
4. Hu et al., *LoRA: Low-Rank Adaptation* (2021) — 4 strony matematyki, którą już znasz.
5. Dokumentacja `mlx-lm` (sekcja LoRA) — gdy siadasz do Maca.

## 5. Ćwiczenia

1. **(obowiązkowe)** Rachunek pamięci LoRA z sekcji 2. Zapisz w notatkach z wnioskiem: 3B czy 8B na Twoim Macu?
2. Zaimplementuj maskę loss na prompt (loss tylko z tokenów odpowiedzi): w `cross_entropy` pomiń pozycje, gdzie `targets == -1`. Dodaj do gradchecka.
3. Napisz BPE: 30 linii Pythona, słownik 300 tokenów na Tiny Shakespeare. Ile tokenów ma tekst? Wytrenuj mikro-GPT z tym tokenizerem (`vocab_size=300`). Loss **nie jest porównywalny** z char-level (inne jednostki) — dlaczego? Porównaj za to wygenerowany tekst.
4. Opcjonalne: LoRA na mikro-GPT (sekcja 2, ćwiczenie praktyczne).

## Co zapamiętać

- Duży LLM = Twój mikro-GPT z podmienionymi klockami (BPE, RoPE, RMSNorm, SwiGLU, GQA) i 30 000× większy. Zasady te same.
- Fine-tuning = ta sama pętla treningowa na Twoich danych, z małym lr. Pełny jest za drogi w pamięci (16 B/param).
- LoRA = zamroź `W`, trenuj `A@B` o niskiej randze; 100–200× mniej pamięci na optymalizator. Napisałbyś to w 15 linii.
- iPad: zrozumienie + inferencja + few-shot. Mac/GPU: LoRA. Z powrotem na iPad: GGUF.

---

**Koniec kursu.** Masz od zera napisany transformer, przetestowany numerycznie autograd, wytrenowany model, przeczytane głowy uwagi, zmierzoną skuteczność few-shot na własnym problemie i policzoną pamięć na LoRA. Czarnej skrzynki już nie ma.
