# Lekcja 8 — iPad jako lab inferencji: lokalny model 3–8B i prompty RCA

**Czas:** 2–3 h · **Cel:** uruchamiasz na iPadzie gotowy model instruct (3–8B, 4-bit), rozumiesz co znaczy "4-bit" i ile pamięci to zajmie, budujesz few-shot prompt z Twoich przykładów "flow + RTT + wniosek", i **mierzysz** (nie "czujesz"), czy model umie RCA bez treningu. Na koniec masz liczbę, która mówi, czy fine-tuning jest w ogóle potrzebny.

To jest najlepsze użycie iPada M5 w całym planie. Trening był po to, żebyś wiedział, co się dzieje w środku tego, co teraz uruchamiasz.

---

## 1. Teoria w 10 minut

### Co to jest "model 3B Q4"

- **3B** = 3 miliardy parametrów. Twój mikro-GPT miał 0.0001B. Ta sama architektura (bloki: attention + MLP + LN + residual), tylko `n_layer≈28–36`, `n_embd≈2048–3072`, `n_head≈16–24`, `block_size` 32k–128k, tokenizer BPE ze 150k tokenów zamiast 65 znaków.
- **Q4 / 4-bit** = każda waga zapisana na 4 bitach zamiast 16/32. Wagi float16 3B = 6 GB; Q4 = ~2 GB. Jakość spada minimalnie (stratność jak w JPEG). Bez kwantyzacji 8B nie zmieściłby się na 12 GB iPadzie.
- **Instruct** = po pretrainingu (zgadywanie następnego tokena, jak Twój) model dostał drugi etap: uczenie na parach "polecenie → odpowiedź" (SFT) i preferencjach ludzi (RLHF/DPO). Dlatego odpowiada na pytania, a nie "kontynuuje tekst". Bazowy model (bez instruct) zachowałby się jak Twój mikro-GPT: dopisałby kolejny incydent.

### Pamięć: co się zmieści

| Model | Q4 na dysku/RAM | iPad 12 GB | iPad 16 GB |
|-------|-----------------|------------|------------|
| Qwen2.5-1.5B / Llama-3.2-1B | ~1 GB | tak, szybko | tak |
| Qwen2.5-3B / Llama-3.2-3B / Phi-3.5-mini (3.8B) | ~2–2.5 GB | **tak (cel)** | tak |
| Qwen2.5-7B / Llama-3.1-8B | ~4.5–5 GB | ryzykownie (iPadOS ubija apki > ~5–6 GB) | tak, zamknij resztę |
| 14B | ~8–9 GB | nie | nie (praktycznie) |

Do RAM dochodzi **KV-cache** (zapamiętane K i V dla każdego tokena kontekstu — to te same K, V z lekcji 4!). Dla 8B i 8k tokenów kontekstu to dodatkowe ~1 GB. Dlatego długi prompt z 20 przykładami kosztuje pamięć i czas.

### Prefill vs generowanie

- **Prefill** = przetworzenie promptu (wszystkie tokeny naraz, jak Twój forward na batchu). Skaluje się z długością promptu. Na M5 jest szybki (tu pomagają Neural Accelerators w GPU).
- **Generowanie** = token po tokenie (jak Twoja pętla `generate`). Każdy token wymaga przeczytania **wszystkich** wag → ograniczone przepustowością pamięci (~150 GB/s na M5). 3B Q4 ≈ 2 GB na token → teoretycznie ~70 tok/s, realnie 20–40. 8B ≈ 10–15 tok/s.

Wniosek praktyczny: długi few-shot prompt jest tani (prefill), długa odpowiedź droga. Każ modelowi odpowiadać krótko i w stałym formacie.

## 2. Krok po kroku

### Krok 2.1 — aplikacja i model

Wybierz jedną (wszystkie darmowe lub z darmowym wariantem, wszystkie działają offline):

| Aplikacja | Silnik | Plusy |
|-----------|--------|-------|
| **PocketPal AI** | llama.cpp (GGUF) | największy wybór modeli (Hugging Face), parametry samplingu, system prompt, eksport rozmów |
| **LLM Farm** | llama.cpp | podobnie, bardziej "techniczne" ustawienia (kontekst, threads) |
| **Locally AI** / **Private LLM** | MLX / własny | używa GPU przez MLX — zwykle szybszy prefill na M5 |
| **Enchanted / Ollama-klient** | zdalny Ollama | tylko jeśli masz Maca w sieci — wtedy to nie jest "iPad-only" |

Pobierz **Qwen2.5-3B-Instruct Q4_K_M** (albo Llama-3.2-3B-Instruct Q4). Na 16 GB dodatkowo **Qwen2.5-7B-Instruct Q4_K_M**. Ustaw kontekst 4096–8192 tokenów.

> ✅ **Sprawdź:** zadaj pytanie "Wyjaśnij w 3 zdaniach, czym różni się utrata pakietów od wysokiego RTT." Odpowiedź w kilka sekund, po polsku, merytorycznie poprawna. Zanotuj tok/s (aplikacja pokazuje).

### Krok 2.2 — zero-shot: czy model w ogóle umie RCA

Wklej **tylko** sekcję SYSTEM z `prompty/rca_fewshot.md` jako system prompt, a jako wiadomość jeden przypadek testowy (bez przykładów). Zrób to dla 5 przypadków (weź z `data/incidents.txt` różne kategorie, usuń linie `wniosek:` i `akcja:`).

Zapisuj odpowiedzi do `moj/rca_wyniki.md`. Oceń wg `prompty/arkusz_oceny.md`.

> ✅ **Sprawdź:** typowo 3B zero-shot: kategoria trafiona w ~50–70%, akcja ogólnikowa ("sprawdź sieć"), format często złamany (dopisuje wstęp, listy). To jest baseline.

### Krok 2.3 — few-shot: 6 przykładów

Wklej cały `prompty/rca_fewshot.md` (SYSTEM + 6 przykładów) i te same 5 przypadków testowych.

> ✅ **Sprawdź:** format prawie zawsze OK, kategoria ~70–90%, akcja konkretniejsza (nazywa urządzenie/site z danych). To jest **in-context learning**: model nie zmienił ani jednej wagi. Attention w każdej warstwie "patrzy" na Twoje przykłady tak samo, jak głowa "początek linii" w Twoim mikro-GPT patrzyła na `\n` — i kopiuje wzorzec. Zobaczyłeś ten mechanizm w lekcji 6.

### Krok 2.4 — Twoje 20 przykładów

Teraz właściwa praca. Zastąp 6 syntetycznych przykładów **20 prawdziwymi** (zanonimizowanymi) z Twojej sieci. Zasady, które wynikają z tego, co wiesz o attention:

1. **Identyczny format każdego przykładu** — model kopiuje strukturę, nie treść. Jedno odstępstwo (np. brak pola LOSS) zostanie skopiowane jako "dozwolone".
2. **Zbalansowane kategorie** — 20 przykładów, 6 kategorii → 3–4 na kategorię. 15 "saturacja" i 1 "routing" = model będzie widział saturację wszędzie (to ten sam efekt, co bigram nauczony na przekrzywionej statystyce).
3. **Przypadki graniczne** — dodaj 2–3 przykłady "niejednoznaczne" z uczciwą odpowiedzią "brakuje metryki X". Inaczej model nigdy nie powie "nie wiem" (lekcja 7, krok 2.5).
4. **Przypadki testowe ≠ przykłady.** 10 testowych trzymasz osobno. Testowanie na przykładach z promptu to test na train (lekcja 1).
5. Liczby podawaj **z baseline'em** (`baseline 8 ms, teraz 310 ms`), bo model rozumie porównanie lepiej niż bezwzględne "310 ms".

Uruchom 10 testowych na 3B z 20 przykładami. Wypełnij arkusz. Jeśli masz 16 GB — powtórz na 7B/8B.

### Krok 2.5 — decyzja

Arkusz daje wynik /40. Kryteria są w `prompty/arkusz_oceny.md`:

- **≥ 32** → few-shot wystarcza. Nie potrzebujesz fine-tuningu. Zbuduj wokół tego narzędzie (skrypt, który wkleja metryki do szablonu i woła model).
- **20–31** → model "prawie umie" — to jest przypadek na LoRA (lekcja 9): 200–500 przykładów w tym samym formacie, trening na Macu, wynik zwykle skacze o 20–30%.
- **< 20** → problem w definicji zadania. Sprawdź, czy dwóch inżynierów z tymi samymi metrykami da tę samą odpowiedź. Jeśli nie — żaden model tego nie zrobi.

To jest najtańsza możliwa weryfikacja hipotezy "LLM pomoże w RCA". Godzina na iPadzie zamiast tygodnia fine-tuningu.

## 3. Ćwiczenia

1. **(obowiązkowe)** Wypełnij arkusz dla: zero-shot, 6 przykładów, Twoje 20 przykładów — na 3B. Zapisz wynik i wniosek w `moj/notatki.md`.
2. Temperatura: to samo co w lekcji 6. Ustaw 0.1 i 1.0 w aplikacji, ten sam prompt 3×. Przy jakiej temperaturze odpowiedzi RCA są stabilne? (Do zadań klasyfikacyjnych: niska.)
3. Usuń z SYSTEM zdanie o "niejednoznaczne". Daj przypadek z RTT wysokim **i** loss wysokim. Czy model przyznaje się do niepewności, czy zgaduje?
4. Zmierz: czas odpowiedzi z 6 vs 20 przykładami. Ile tokenów ma prompt (aplikacja pokazuje)? Ile to KB tekstu (~4 znaki/token)? Porównaj z `block_size` Twojego mikro-GPT.
5. Poproś model o **uzasadnienie** przed wnioskiem ("najpierw wypisz, które metryki są poza normą, potem WNIOSEK"). Czy trafność rośnie? To jest chain-of-thought — model "myśli" tokenami, bo tylko tak umie (każdy token to jeden forward).

## Co zapamiętać

- 3B Q4 = ta sama architektura co Twój mikro-GPT, 30 000× więcej parametrów, wagi skompresowane do 4 bitów. Na 12 GB celuj w 3–4B, na 16 GB do 8B.
- Prefill tani, generowanie drogie → długi prompt OK, krótka odpowiedź w stałym formacie.
- Few-shot = in-context learning przez attention. Format i balans przykładów ważniejsze niż ich liczba.
- Mierz arkuszem, nie wrażeniem. Wynik arkusza decyduje, czy w ogóle potrzebujesz fine-tuningu.

➡️ [Lekcja 9 — Most dalej: BPE, fine-tuning, LoRA, co na Maca](09_co_dalej.md)
