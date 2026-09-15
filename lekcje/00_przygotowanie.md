# Lekcja 0 — Przygotowanie iPada i sprawdzenie sprzętu

**Czas:** ~1 h · **Cel:** masz działający Python z NumPy, dane treningowe w folderze, wiesz, ile Twój iPad realnie udźwignie, i wiesz, jak pracować z tym kursem.

---

## 1. Co będziemy robić i dlaczego to działa na iPadzie

Model językowy (LLM) to program, który dostaje kawałek tekstu i zgaduje **następny znak** (albo słowo).
Cały "GPT" to tylko bardzo dobry zgadywacz następnego znaku, uruchamiany w pętli.

Zbudujemy taki zgadywacz od zera. Nasz będzie **mikro**: ~100 000 parametrów (liczb, które model "uczy się"), gdy ChatGPT ma ich setki miliardów. Ale **mechanika jest identyczna** — ta sama matematyka, te same elementy (attention, residual, LayerNorm), ta sama krzywa uczenia. Różnica to skala.

Dlaczego mikro wystarczy na iPadzie:

| Co | Ile pamięci | Ile czasu na iPadzie M5 |
|----|-------------|--------------------------|
| Nasz mikro-GPT (100k parametrów) | ~2 MB | trening 2–8 min |
| Większy mikro-GPT (800k) | ~13 MB | 15–40 min |
| Fine-tuning modelu 3B (LoRA) | 6–12 GB + narzędzia, których w iPadOS nie ma | nie na iPadzie |
| **Uruchomienie** gotowego modelu 3–8B w 4-bit | 2–5 GB | działa, lekcja 8 |

Wąskie gardło to nie procesor (M5 jest szybki), tylko **iPadOS**: nie ma PyTorcha, nie ma MLX dla Pythona, procesy giną po zgaszeniu ekranu. Dlatego piszemy wszystko w czystym NumPy, które na iPadzie jest od lat.

## 2. Zainstaluj aplikację z Pythonem

Wybierz **jedną** (możesz mieć kilka; kurs zakłada a-Shell, ale wszystko działa wszędzie):

| Aplikacja | Cena | Plusy | Minusy |
|-----------|------|-------|--------|
| **a-Shell** (polecana) | darmowa | prawdziwy terminal, `python3`, NumPy, matplotlib, `curl`, `lg2` (git), dostęp do folderów w Plikach | wykresy oglądasz jako pliki PNG (`open plik.png`) |
| **Carnets** | darmowa | Jupyter: kod + wykres w jednym miejscu | wolniejsze uruchamianie, mniej "terminalowo" |
| **Pythonista 3** | ~50 zł | najlepszy edytor, NumPy+matplotlib wbudowane, wykres w konsoli | brak terminala; git przez osobną apkę |
| **Juno** | ~80 zł | ładny Jupyter | płatny, to samo co Carnets |

> Uwaga a-Shell: zainstaluj **a-Shell**, nie **a-Shell mini** — mini nie ma NumPy.

## 3. Pobierz kurs na iPada

**Sposób A — a-Shell + git (`lg2`)**

```
cd ~/Documents
lg2 clone https://github.com/TadeuszWolfGang/Learn-LLM.git
cd Learn-LLM
ls
```

**Sposób B — aplikacja Working Copy** (darmowa do klonowania): sklonuj repo, potem w a-Shell wpisz `pickFolder` i wybierz folder Working Copy → Learn-LLM. a-Shell "wchodzi" do tego folderu.

**Sposób C — bez gita:** na github.com kliknij *Code → Download ZIP*, rozpakuj w aplikacji Pliki, w a-Shell `pickFolder` i wskaż folder.

Zrób sobie folder na własny kod:

```
mkdir moj
```

Wszystko, co piszesz w kursie, ląduje w `moj/`. `rozwiazania/` to ściąga.

## 4. Pobierz dane: Tiny Shakespeare

To 1 MB tekstu (wszystkie sztuki Szekspira sklejone). Klasyczny zbiór do mikro-modeli, bo jest mały, a ma strukturę (imiona postaci, dwukropki, wersy).

```
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt
head -20 data/input.txt
```

Powinieneś zobaczyć:

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

(W Pythonista/Carnets: pobierz plik przez Safari i przenieś do folderu, albo uruchom w Pythonie
`import urllib.request; urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "data/input.txt")`.)

## 5. Pierwszy program: sprawdź, że NumPy żyje

Utwórz plik `moj/test.py` (w a-Shell: `vim moj/test.py` albo edytuj w aplikacji Pliki / dowolnym edytorze, np. Textastic, Runestone):

```python
import numpy as np, time

a = np.random.randn(1024, 1024).astype(np.float32)   # macierz 1024x1024 losowych liczb
a @ a                                                # rozgrzewka
t = time.time()
for _ in range(5):
    a @ a                                            # mnożenie macierzy - to robi 90% pracy w LLM
gflops = 5 * 2 * 1024**3 / (time.time() - t) / 1e9
print(f"NumPy {np.__version__}: {gflops:.0f} GFLOP/s")
```

Uruchom: `python3 moj/test.py`. Zobaczysz coś w stylu `NumPy 1.26: 40 GFLOP/s` — liczba między 10 a 200 jest OK. To znaczy, że Twój iPad robi dziesiątki miliardów mnożeń na sekundę. Mikro-GPT potrzebuje ~1 miliarda na krok.

## 6. Zmierz, ile potrwa trening

Uruchom gotowy skrypt z rozwiązań (jedyny raz w kursie, gdy odpalasz cudzy kod przed napisaniem własnego — chodzi tylko o pomiar):

```
cd rozwiazania
python3 00_check_env.py
cd ..
```

Zobaczysz m.in.:

```
krok GPT : 120 ms  (model 112,512 parametrów, batch 32x64)
Szacunek : lekcja 5 (2000 kroków) ≈ 4.0 min
```

Zapisz sobie tę liczbę. Jeśli krok trwa > 500 ms, w lekcji 5 zmniejsz `batch_size` do 16.

## 7. Ustawienia iPada na czas treningu

- Ustawienia → Ekran i jasność → **Blokada automatyczna: Nigdy** (przywróć po kursie).
- Podłącz zasilanie. iPad nie ma wentylatora; po kilku minutach pełnego obciążenia zwalnia (throttling). Dla naszych rozmiarów to bez znaczenia.
- Nie przełączaj się na inne aplikacje w trakcie treningu (iPadOS może uśpić proces). Split View z notatkami obok jest OK, dopóki a-Shell jest widoczny.
- Skrypty w kursie zapisują **checkpoint** co 100 kroków. Jeśli proces zginie, wznawiasz od checkpointu (lekcja 5).

## 8. Jak pracować z lekcjami

- **Przepisuj kod ręcznie.** Wklejanie omija mózg. Przepisując, zatrzymasz się przy każdej linijce, której nie rozumiesz — i o to chodzi.
- Po każdym kawałku kodu jest ramka **"Sprawdź"** z tym, co masz zobaczyć. Jeśli widzisz coś innego — stop, szukaj błędu, dopiero potem czytaj dalej.
- Sekcja **Ćwiczenia** na końcu: pierwsze zadanie zawsze zrób. Reszta to bonus.
- Utknąłeś > 20 min? Otwórz odpowiedni plik w `rozwiazania/`, porównaj **tylko ten fragment**, wróć do własnego kodu.
- Zapisuj notatki (co zobaczyłeś, co Cię zaskoczyło). Na koniec kursu będziesz miał własny "podręcznik".

## Ćwiczenia

1. Sprawdź, ile jest różnych znaków w `data/input.txt` (podpowiedź: `len(set(open("data/input.txt").read()))`). Zapamiętaj tę liczbę — to będzie rozmiar "słownika" modelu.
2. Ile razy w tekście występuje słowo "ROMEO"? A "\n\n" (pusta linia)? Zgadnij, do czego model może wykorzystać puste linie.
3. Zapisz w `moj/notatki.md`: wynik GFLOP/s, czas kroku z punktu 6, nazwę aplikacji, której używasz.

## Co zapamiętać

- LLM = zgadywacz następnego znaku uruchamiany w pętli. Mikro i duży różnią się skalą, nie zasadą.
- Na iPadzie ogranicza Cię system (brak PyTorch, brak tła), nie procesor. Czysty NumPy omija problem.
- Kod przepisujesz sam; `rozwiazania/` służy do porównania, nie do uruchamiania.

➡️ [Lekcja 1 — Tekst → liczby. Bigram i pojęcie "loss"](01_tekst_i_bigram.md)
