# Lekcja 7 — Trening na własnych danych (incydenty sieciowe)

**Czas:** 1–2 h · **Cel:** trenujesz swój mikro-GPT na tekście z Twojej domeny (opisy incydentów: flow, RTT, wniosek). Widzisz dokładnie, **co** model łapie po kilku minutach treningu (format, słownictwo, korelacje), a **czego nigdy** nie złapie (przyczynowość, liczby jako liczby). To kalibruje oczekiwania przed lekcją 8 i 9.

---

## 1. Teoria w 5 minut: czego uczy się model językowy

Model minimalizuje zaskoczenie następnym znakiem. Nauczy się więc **wszystkiego, co jest regularne w tekście**:

- format (`[HH:MM] INCYDENT nnnn | A -> B | app proto`),
- słownictwo domeny (`retransmisje`, `bufferbloat`, `SFP`),
- **korelacje**: po linii `loss: 6%` statystycznie częściej idzie `problem L1/L2` niż `saturacja`.

Nie nauczy się:

- **liczb jako wielkości**: `310 ms` to dla niego 6 znaków, nie "dużo". Nie porówna `310 > 8`.
- **przyczynowości**: zna kolejność słów, nie mechanizm.
- niczego, czego nie ma w danych — 20 Twoich incydentów to za mało, żeby cokolwiek wyszło ponad format.

Prawdziwe LLM-y "wydają się" rozumieć liczby i przyczyny, bo widziały biliony tokenów, w tym miliony podręczników, gdzie te zależności są opisane słowami. Mechanizm jest ten sam, co w Twoim mikro-GPT. Skala inna.

## 2. Krok po kroku

### Krok 2.1 — zbuduj korpus

Char-level model potrzebuje **minimum ~200 KB** tekstu, żeby wyjść poza bełkot. 20 ręcznie napisanych incydentów to ~5 KB. Dwie drogi:

**(a) Generator z szablonów** (na dziś): `python3 rozwiazania/07_gen_incidents.py` tworzy `data/incidents.txt` (~480 KB, 1500 syntetycznych incydentów w 6 kategoriach: opóźnienie WAN, straty L1/L2, saturacja, aplikacja/DNS, routing asymetryczny, norma). Przejrzyj `head -30 data/incidents.txt`. Otwórz `07_gen_incidents.py` i **dopisz własną kategorię** (np. "MTU/fragmentacja" albo "VPN rekey") — to ćwiczenie w myśleniu, jakie sygnały ją odróżniają.

**(b) Własne dane** (docelowo): eksport z Twojego systemu monitoringu (opisy ticketów, notatki post-mortem, komentarze z RCA), zanonimizowany (IP, nazwy hostów). Jeden incydent = jeden blok, blok od bloku oddzielony pustą linią, **zawsze te same pola w tej samej kolejności**. Konsekwencja formatu > ilość.

### Krok 2.2 — trening

Twój `05_train.py` ma `load_dataset()` z domyślną ścieżką. Dodaj parametr ścieżki (w rozwiązaniach: `--data`). Słownik zbuduje się z nowego pliku (~75 znaków — są cyfry, `[`, `|`, `>`).

```
python3 moj/05_train.py --data data/incidents.txt --out out_inc
```

(albo zmień ścieżkę w kodzie). 1000 kroków wystarczy — korpus jest bardzo regularny.

> ✅ **Sprawdź:** loss spada **dużo szybciej i niżej** niż na Szekspirze: po 300 krokach val ~0.6, po 1000 ~0.36 (Szekspir po 1000: 1.9). Nie dlatego, że model jest mądrzejszy — dane są przewidywalne (szablony). Perplexity ~1.4 = model waha się średnio między jednym a dwoma znakami. To pierwszy sygnał: **niski loss ≠ inteligencja, niski loss = przewidywalne dane.**

### Krok 2.3 — oglądaj, co wygenerował

```
python3 moj/06_sample.py out_inc/best.npz --prompt "[10:15] INCYDENT" --temp 0.5 --n 500
```

(Tokenizer musi być zbudowany z **tego samego** pliku, na którym trenowałeś — inaczej numery znaków się nie zgadzają. W rozwiązaniach: `--data data/incidents.txt` także przy próbkowaniu.)

Przeczytaj uważnie 3–4 wygenerowane incydenty i **zaznacz w notatkach**:

1. Czy format jest poprawny (kolejność pól, separatory)? — *będzie*.
2. Czy IP wyglądają jak IP? — *będą*, ale losowe.
3. Czy `wniosek` pasuje do metryk powyżej? Np. po `loss: 8%` jest `L1/L2`? — *często tak*: model złapał korelację "słowo loss z dużą liczbą → L1/L2". Ale spróbuj promptu z `loss: 0%` i `RTT: teraz 900 ms` — czasem nadal napisze L1/L2. Bo nie porównuje liczb; dopasowuje wzorzec tekstowy.
4. Czy `akcja` odnosi się do **tego samego** site'u/aplikacji co nagłówek? — *rzadko*. Kontekst 64 znaków nie sięga od nagłówka do akcji (blok ma ~350 znaków). Model "zapomniał" nagłówek. To jest **bezpośrednio** ograniczenie `block_size`.

Przykład z modelu wytrenowanego 1000 kroków (T=0.5):

```
[0:40] INCYDENT 2705 | AZR-WE -> AZR-WE | backup TCP/22
flow: 10.24.9.190:18928 -> 10.212.47.23:5060
RTT: 18 ms w jedna strone, 378 ms w druga
TTL: rozne wartosci na sciezkach tam/powrot (50 vs 46)
wniosek: routing asymetryczny -> powrot idzie inna, dluzsza sciezka
akcja: sprawdz tablice routingu i BGP preferencje w GDA-OFF
```

Format i para "RTT asymetryczny → routing asymetryczny" bezbłędne (to jest w zasięgu 64 znaków). Ale: godzina `0:40` zamiast `00:40`, port `5060` przy `TCP/22`, a akcja wskazuje `GDA-OFF`, choć incydent jest w `AZR-WE` — wszystko, co wymaga spojrzenia dalej niż 64 znaki wstecz, jest losowe.

### Krok 2.4 — eksperyment: dłuższy kontekst

Wytrenuj z `block_size=256` (i `n_embd=64`, `n_layer=2` — trening ~4× wolniejszy, bo attention T² ). Powtórz punkt 4.

> ✅ **Sprawdź:** akcja zaczyna zgadzać się z nagłówkiem (site, aplikacja). Model "widzi" cały blok. To jest dokładnie to, co attention daje, a MLP z lekcji 3 nie mógłby — i dlaczego kontekst jest tak cennym zasobem w dużych modelach.

### Krok 2.5 — test: prompt spoza rozkładu

Daj prompt z metryką, której **nie ma w szablonach**, np. `jitter: 45 ms`. Model dopisze coś "w stylu" korpusu, ignorując nowe pole. Nie ma mechanizmu "nie wiem". To jest halucynacja w miniaturze — ten sam mechanizm co w dużych modelach.

## 3. Wnioski dla Twojego use-case (RTT / RCA)

| Chcesz | Mikro-GPT na iPadzie | Gotowy 3–8B (lekcja 8) | LoRA 3–8B na Macu (lekcja 9) |
|--------|----------------------|------------------------|------------------------------|
| Format raportu RCA | tak, po 1000 krokach | tak, z 1 przykładu w prompcie | tak |
| Słownictwo domeny | tak | ma już z pretrainingu | tak, dokładniej Twoje |
| "loss wysoki, RTT niski → L1/L2" (reguła) | korelacja tekstowa, zawodna | tak, rozumie liczby i zna sieci | tak, stabilniej |
| Nowa reguła, której nie ma w internecie | nie | przez few-shot, jeśli prosta | tak, to jest cel LoRA |
| Sensowna akcja dla konkretnego site'u | nie | tak, jeśli podasz kontekst | tak |

Wniosek: mikro-model **nie jest** narzędziem do RCA i nigdy nie będzie. Jest narzędziem do zrozumienia, **co** robi model, który za chwilę uruchomisz w lekcji 8 — i dlaczego dobre przykłady w prompcie działają (to "trening w kontekście" na tej samej mechanice attention, którą napisałeś).

## 4. Ćwiczenia

1. **(obowiązkowe)** Wypisz 3 rzeczy, które model zrobił dobrze i 3, które źle, z konkretnymi cytatami z wygenerowanego tekstu.
2. Zbuduj korpus z mieszanki: 50% Szekspir, 50% incydenty (`cat data/input.txt data/incidents.txt > data/mix.txt`). Co generuje model po prompcie `ROMEO:`, a co po `[10:15] INCYDENT`? Czy "przecieka" styl między domenami?
3. Napisz ręcznie 5 prawdziwych (zanonimizowanych) incydentów w formacie z generatora. Doklej ×50 (powtórz plik 50 razy) do korpusu. Czy model je "zapamięta"? Sprawdź, czy generuje je dosłownie — to jest memoryzacja, ten sam mechanizm, przez który duże modele wyciekają dane treningowe.
4. Policz: ile znaków kontekstu potrzebujesz, żeby model widział cały Twój prawdziwy incydent? Ile to tokenów BPE (~4 znaki/token)? To jest liczba, którą podasz jako `max context` w lekcji 8.

## Co zapamiętać

- Model uczy się regularności tekstu: format, słowa, korelacje. Nie liczb jako liczb, nie przyczyn.
- Niski loss na przewidywalnych danych nie znaczy nic.
- `block_size` to twarda granica "pamięci" modelu — widać to natychmiast na danych z długą strukturą.
- Halucynacja = model dopisuje najbardziej prawdopodobną kontynuację, także gdy nie ma podstaw. Nie ma stanu "nie wiem".

➡️ [Lekcja 8 — iPad jako lab inferencji: lokalny 3–8B i prompty RCA](08_inferencja_rca.md)
