# Learn-LLM — zbuduj własny mikro-LLM na iPadzie, krok po kroku

Kurs dla laika. Po jego przejściu **samodzielnie** napiszesz od zera, w czystym Pythonie + NumPy,
mały model językowy typu GPT (tokenizer → autograd → attention → transformer → trening → generowanie),
wytrenujesz go na iPadzie i zrozumiesz, co się dzieje w środku. Bez PyTorcha, bez MLX, bez Maca.

Druga część pokaże Ci, jak używać iPada jako laboratorium do testowania gotowych modeli
(3–8B) na Twoich danych sieciowych (RTT, flow, RCA) — i dlaczego fine-tuning to już robota na inną maszynę.

## Jak korzystać z tego kursu

1. Lekcje czytasz **po kolei** — każda zakłada, że masz kod z poprzedniej.
2. W każdej lekcji **przepisujesz kod sam** (nie kopiuj-wklej, serio: przepisywanie = uczenie).
   Kod jest podany w małych kawałkach, każdy z wyjaśnieniem i z tym, co masz zobaczyć po uruchomieniu.
3. Folder `rozwiazania/` to ściąga. Zaglądasz tam **dopiero gdy utkniesz** albo żeby porównać po skończeniu.
4. Swój kod trzymasz w folderze `moj/` (jest w `.gitignore` tylko dla wyników; kod możesz commitować).
5. Każda lekcja kończy się ćwiczeniami. Zrób co najmniej pierwsze z każdej — to one budują samodzielność.

## Plan lekcji

| # | Lekcja | Czas | Co umiesz po niej |
|---|--------|------|-------------------|
| 0 | [Przygotowanie iPada i sprawdzenie sprzętu](lekcje/00_przygotowanie.md) | 1 h | Masz działający Python+NumPy, dane, wiesz co iPad M5 udźwignie |
| 1 | [Tekst → liczby. Bigram i pojęcie "loss"](lekcje/01_tekst_i_bigram.md) | 2 h | Tokenizer, train/val, cross-entropy, pierwszy generator tekstu |
| 2 | [Jak model się uczy: gradient i własny autograd](lekcje/02_autograd.md) | 4–5 h | Piszesz silnik automatycznego różniczkowania i testujesz go numerycznie |
| 3 | [Embedding i sieć neuronowa z kontekstem (MLP)](lekcje/03_mlp.md) | 2 h | Embedding, warstwa ukryta, Adam, pierwszy overfitting na własne oczy |
| 4 | [Attention na piechotę: Q, K, V](lekcje/04_attention.md) | 2–3 h | Liczysz attention ręcznie, rozumiesz maskę, softmax i multi-head |
| 5 | [Składamy GPT i trenujemy](lekcje/05_gpt.md) | 3–4 h | LayerNorm, residual, blok transformera, pętla treningowa, checkpointy, wykres loss |
| 6 | [Eksperymenty: overfitting, ablacje, temperatura](lekcje/06_eksperymenty.md) | 2–3 h | Umiesz "czytać" krzywą loss i wiesz, po co jest każdy element modelu |
| 7 | [Trening na własnych danych (incydenty sieciowe)](lekcje/07_wlasne_dane.md) | 1–2 h | Wiesz, czego mikro-model się nauczy, a czego nigdy — i dlaczego |
| 8 | [iPad jako lab inferencji: lokalny 3–8B i prompty RCA](lekcje/08_inferencja_rca.md) | 2–3 h | Odpalasz Qwen/Llama lokalnie, budujesz few-shot na swoich przykładach RTT, oceniasz wynik |
| 9 | [Most dalej: BPE, fine-tuning, LoRA, co na Maca](lekcje/09_co_dalej.md) | 1 h | Rozumiesz, jak Twój mikro-GPT ma się do prawdziwych LLM i co robić dalej |

Razem ~20–25 h. Realistycznie: tydzień po 3 h dziennie albo dwa intensywne weekendy.
Lekcje 0–6 są rdzeniem ("przestań traktować transformer jak czarną skrzynkę"). 7–9 to część praktyczna pod Twój use-case.

## Struktura repo

```
lekcje/          <- kurs, czytaj po kolei
rozwiazania/     <- gotowy, przetestowany kod (ściąga; ~600 linii razem)
  autograd.py      silnik autograd na NumPy (lekcja 2)
  data.py          tokenizer + batche (lekcja 1)
  model.py         mikro-GPT (lekcje 4-5)
  0X_*.py          skrypty do każdej lekcji
data/            <- Tiny Shakespeare + przykładowe incydenty sieciowe
prompty/         <- szablon few-shot do RCA (lekcja 8)
moj/             <- tu piszesz swój kod (tworzysz sam)
```

## Korekty do założeń wyjściowych (co było nie tak w pierwotnym planie)

- **Pamięć iPada M5.** Nie ma wersji 18 GB. iPad Pro M5: 12 GB (256/512 GB) lub 16 GB (1/2 TB). Dla mikro-modelu to bez znaczenia (potrzebujesz < 100 MB). Ma znaczenie dla lekcji 8: na 12 GB celuj w modele 3–4B 4-bit, 8B 4-bit tylko na 16 GB.
- **micrograd.py Karpathy'ego to zły wybór.** Micrograd liczy pochodne na *pojedynczych liczbach*. Transformer o 100k+ parametrów to miliony obiektów Pythona na jeden krok — godziny na jeden batch. Zamiast tego w lekcji 2 piszesz **ten sam pomysł, ale na całych macierzach NumPy** (~250 linii). Trening trwa minuty.
- **"0.5–2M parametrów, ~80 MB z Adamem"** — rachunek: parametry × 4 bajty × 4 kopie (wagi, gradient, m, v) = 16 bajtów/parametr. 2M → 32 MB. Domyślny model z kursu ma ~110k parametrów (~2 MB). Pamięć nie jest ograniczeniem, jest nim *czas na CPU*.
- **Realny czas kroku w NumPy na CPU**: ~50–250 ms dla modelu 2 warstwy × 64 wymiary × kontekst 64 (zależy od NumPy na iPadzie). 2000 kroków = 2–8 min. Model 4×128 to ~4× dłużej. Lekcja 0 mierzy to na Twoim iPadzie.
- **"Nie uruchomisz w tle na noc"** — prawda, ale nie musisz: wszystko w kursie trwa < 30 min. Ekran włączony + zasilanie + Auto-Lock "Nigdy" wystarczy.
- **Swift + MLX (Opcja A)** wymaga Swift Playgrounds i pakietów, jest gorsze dydaktycznie (mniej widzisz). Kurs jest w 100% Opcja B: Python + NumPy.
- **Czego brakowało w pierwotnym planie** (dodane): test gradientów numerycznych (bez tego nie wiesz, czy Twój backward jest poprawny), bigram i MLP *przed* transformerem (Karpathy'ego ścieżka — attention rozumie się tylko wiedząc, co MLP robi źle), positional embeddings, LR warmup + gradient clipping (bez tego mikro-transformer potrafi nie ruszyć), temperatura/top-k, ablacje (usuwasz residual → widzisz po co jest), oraz konkretna metoda oceny promptów RCA w lekcji 8.

## Wymagania

iPad z iPadOS 17+ (M-cokolwiek; M5 to nadmiar mocy) i jedna z aplikacji: **a-Shell** (darmowa, polecana),
**Carnets** (darmowy Jupyter), **Pythonista** (płatna) lub **Juno** (płatny Jupyter). Szczegóły w lekcji 0.
