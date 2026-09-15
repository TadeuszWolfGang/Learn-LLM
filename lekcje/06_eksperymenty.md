# Lekcja 6 — Eksperymenty: overfitting, ablacje, temperatura, głowy uwagi

**Czas:** 2–3 h · **Cel:** przestajesz wierzyć na słowo, po co jest każdy element modelu — **wyłączasz go i patrzysz**. Uczysz się czytać krzywą loss jak lekarz EKG. Oglądasz, czego nauczyły się głowy uwagi. Rozumiesz temperaturę i top-k. Po tej lekcji, gdy ktoś powie "residual connections stabilizują trening", odpowiesz: "wiem, widziałem".

Wszystkie eksperymenty robisz na swoim kodzie z lekcji 5. Każdy to 600 kroków (1–3 min). Wyniki zapisuj w tabeli w notatkach.

---

## 1. Jak czytać krzywą loss

Cztery obrazki, które musisz rozpoznawać:

```
 loss                       loss                      loss                      loss
  |\                         |\                        |\                        |~~~~~~~~~
  | \__ train                | \_____ train            | \      train           |
  |  \___ val                |  \                      |  \____                 |   (nie rusza)
  |     ‾‾‾‾                 |   \‾‾‾‾‾‾ val (rośnie)  |       ‾‾‾‾‾‾ oba stoją |
  +---------- krok           +---------- krok          +---------- krok         +---------- krok
   ZDROWY                     OVERFITTING               UNDERFITTING             ZEPSUTE
   val ≈ train, oba spadają   train↓ val↑               oba stanęły wysoko       lr / init / bug
   → trenuj dłużej / większy  → więcej danych, mniejszy  → większy model, dłużej,  → sprawdź gradcheck,
     model                      model, dropout, wd        wyższy lr                 lr, warmup
```

Reguła: **patrz na val, decyduj po różnicy train–val.**

## 2. Eksperymenty (zrób wszystkie — każdy to jedna zmiana w kodzie)

Tabela do notatek (bazowy = lekcja 5, 2 warstwy × 64, 600 kroków):

| # | Eksperyment | val @600 | Co widać w tekście | Wniosek |
|---|-------------|----------|--------------------|---------|
| 0 | bazowy | ~2.07 | | |
| 1 | overfitting na życzenie | | | |
| 2 | bez residual (4 warstwy) | | | |
| 3 | 4 warstwy z residual | | | |
| 4 | bez pos_emb | | | |
| 5 | bez LayerNorm | | | |
| 6 | dropout 0.2 | | | |
| 7 | block_size 16 | | | |

### E1 — overfitting na życzenie

Obetnij dane treningowe do 0.5% (`train = train[:5000]`). 600 kroków.

> ✅ **Sprawdź:** train 4.2 → 2.3 → 1.5 → 0.7 → **0.5**. Val: 4.2 → 2.8 → **3.1 → 3.9 → 4.9**. Val kończy **gorzej niż zgadywanie** (4.17)! Model wykuł 5000 znaków na pamięć i jest pewny siebie w błędach. Wygeneruj tekst: to będą dosłowne cytaty z tych 5000 znaków (sprawdź `grep`-em).
> Teraz dodaj `dropout=0.2` i `weight_decay=1.0`. Val nadal rośnie, ale wolniej i później. Regularyzacja **opóźnia** overfitting, nie leczy — lekiem są dane.

### E2/E3 — residual: wyłącz i zobacz

W `Block.__call__` zamień `x = x + self.attn(...)` na `x = self.attn(...)` (i tak samo dla mlp). Uruchom z `n_layer=4`. Potem to samo **z** residualem.

> ✅ **Sprawdź:** bez residual po 100 krokach loss ~3.3 (z residualem: ~2.5), po 600 krokach nadal ~2.6–2.9 i stoi — gradient nie dociera do dolnych warstw. Z residualem 4 warstwy: ~2.0, lepiej niż 2 warstwy. **Residual = różnica między "głębiej = lepiej" a "głębiej = nie da się wytrenować".** Przy 8+ warstwach bez residuala model w ogóle nie ruszy z 4.17.

### E4 — bez pozycji

W `GPT.__call__` usuń `+ self.pos_emb[np.arange(T)]`.

> ✅ **Sprawdź:** loss wyraźnie gorszy (~2.3 vs 2.07), a tekst: słowa "od środka", np. spacje w dziwnych miejscach, litery w losowej kolejności w słowie. Model widzi **zbiór** poprzednich znaków, nie **ciąg**. To jest lekcja 4, ćwiczenie 5, na żywo.

### E5 — bez LayerNorm

Zamień `self.ln1(x)` na `x` (i ln2, ln_f). Uruchom 3 razy z różnymi `seed` (1, 2, 3).

> ✅ **Sprawdź:** czasem trenuje (trochę gorzej), czasem `nan` po kilkuset krokach, czasem stoi. **LayerNorm zamienia loterię w powtarzalność.** Bez niego "działa u mnie, nie działa u ciebie" — to jest praktyczny powód, dla którego jest w każdym LLM.

### E6 — dropout na pełnych danych

`dropout=0.2`, pełne dane, 600 kroków.

> ✅ **Sprawdź:** val **gorszy** niż bazowy (~2.15). Na 1 MB danych i 110k parametrów nie ma overfittingu, więc regularyzacja tylko przeszkadza. Dropout ma sens, gdy train ≪ val. Nie dodawaj regularyzacji "na zapas".

### E7 — kontekst 16 zamiast 64

`block_size=16` (i `get_batch(..., 16, ...)`).

> ✅ **Sprawdź:** val ~2.3. Krótszy kontekst = mniej informacji na każdą predykcję. Ale trening jest 3× szybszy (attention T²). To jest kompromis, który skalujesz w lekcji 8 do "ile tokenów kontekstu dać modelowi 3B".

## 3. Temperatura i top-k (sampling)

Z lekcji 1 wiesz, że losujemy z rozkładu. **Temperatura** dzieli logity przed softmaxem: `p = softmax(logits / T)`.

- `T → 0`: rozkład "wyostrza się" do argmax → tekst poprawny, ale powtarzalny (`the shall the shall the`).
- `T = 1`: rozkład taki, jakiego nauczył się model.
- `T > 1`: spłaszczenie → więcej "kreatywności" = więcej bzdur.

**Top-k**: zostaw tylko k najbardziej prawdopodobnych, resztę wyzeruj. Odcina "ogon" głupich znaków, nie spłaszczając reszty.

Uruchom swoje `generate` z `out/best.npz` (z lekcji 5) dla T = 0.3, 0.8, 1.5 i top_k = 5:

> ✅ **Sprawdź:**
> - T=0.3: `I must we shall for me the world the come to with the such ere the be for the be...` — same częste słowa, zapętlone.
> - T=0.8: zdrowy Szekspir-bełkot ze strukturą.
> - T=1.5: `Ourselvy Gell as but aom, Cto beart. First gyor: I diry yath-end` — rozpad.
> - top_k=5, T=0.8: podobne do T=0.8, mniej "literówek".

Do zadań typu klasyfikacja/RCA (lekcja 8): niska temperatura (0–0.3). Do generowania pomysłów: 0.7–1.0.

## 4. Co widzą głowy uwagi

Model zapisuje `self.last_att` w każdej warstwie. Przepuść przez wytrenowany model tekst `"First Citizen:\nBefore we proceed any"` i narysuj `last_att[0, h]` (macierz T×T) dla każdej warstwy i głowy (8 obrazków; w rozwiązaniach: `07_visualize_attention.py`).

> ✅ **Sprawdź — typowe znaleziska w modelu 2×4** (Twoje mogą się różnić numerami głów, ale wzorce będą):
> - **Warstwa 0, jedna głowa: ciemna przekątna tuż pod główną** = "patrz na poprzedni znak". Bigram wbudowany w attention.
> - **Warstwa 0, inne głowy: bloki od ostatniej spacji** = "patrz na początek bieżącego słowa". Model wie, w którym słowie jest.
> - **Warstwa 1: pionowe kolumny na `\n` lub `:`** = "patrz na początek linii / imię mówiącego". To dlatego generuje strukturę `IMIĘ:\nwers`.
> - Warstwa 1 patrzy **dalej wstecz** (średnio 4–7 znaków) niż warstwa 0 (1–2 znaki). Dolne warstwy = lokalne wzorce, górne = szerszy kontekst. W dużych modelach to samo, tylko 30 warstw.

Policz dla każdej głowy średnią odległość wstecz: `(A * dist).sum(1).mean()`, gdzie `dist[i,j] = i - j`. Wpisz do notatek.

To jest **jedyny moment w kursie, gdy zaglądasz do środka wytrenowanej sieci i widzisz coś interpretowalnego.** W dużych modelach robi to dziedzina "mechanistic interpretability" — tymi samymi narzędziami.

## 5. Skalowanie (jeśli masz 30–40 min)

`n_layer=4, n_embd=128, n_head=4, STEPS=3000` (~800k parametrów).

> ✅ **Sprawdź:** val ~1.5. Tekst: całe poprawne frazy, mniej literówek. 7× więcej parametrów, ~10× więcej obliczeń, loss lepszy o 0.25. To jest "scaling law" w miniaturze: każdy kolejny stały postęp kosztuje wykładniczo więcej. Dokładnie dlatego GPT-4 kosztował 100M$, a nie 10M$.

## 6. Ćwiczenia

1. **(obowiązkowe)** Wypełnij tabelę z sekcji 2 (co najmniej E1–E4) i zapisz jedno zdanie wniosku przy każdym wierszu.
2. Znajdź `lr`, przy którym bazowy model wybucha (`nan`) i taki, przy którym po 600 krokach nie zejdzie poniżej 3.0. Zapisz "okno" działających `lr`. Potem wyłącz warmup i sprawdź, czy okno się zwęża.
3. Zmień liczbę głów: 1, 2, 8 (przy `n_embd=64`). Czy więcej głów = lepiej? Kiedy `hs = n_embd/n_head` robi się za małe?
4. Zaimplementuj "greedy" (argmax) w `generate`. Po ilu znakach model wpada w pętlę?
5. Głowa "poprzedni znak" z sekcji 4: co się stanie, gdy w wytrenowanym modelu **wyzerujesz** jej wagi (`qkv.w` odpowiadające tej głowie)? O ile wzrośnie val? A gdy wyzerujesz losową inną głowę? (To jest "ablacja głowy" — standardowe narzędzie interpretowalności.)

## Co zapamiętać

- Krzywa loss: decyduje różnica train–val. Overfitting = val rośnie. Underfitting = oba stoją wysoko.
- Residual: bez niego głębokie sieci się nie uczą. LayerNorm: bez niego trening to loteria. Pos_emb: bez niego model widzi worek znaków.
- Regularyzacja tylko wtedy, gdy jest overfitting. Nie "na zapas".
- Temperatura: niska do decyzji, średnia do tekstu, wysoka do chaosu. Top-k odcina ogon.
- Głowy uwagi uczą się interpretowalnych wzorców: poprzedni znak, początek słowa, początek linii.

➡️ [Lekcja 7 — Trening na własnych danych](07_wlasne_dane.md)
