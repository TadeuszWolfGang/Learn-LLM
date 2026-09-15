# Arkusz oceny promptów RCA (lekcja 8)

Dla każdego modelu i wariantu promptu: 10 testowych przypadków (nie z przykładów!), oceń 0/1/2.

| # | Przypadek (skrót) | Prawidłowa kategoria | Model odpowiedział | Kategoria OK? (0/1) | Akcja sensowna? (0/1/2) | Format OK? (0/1) | Uwagi |
|---|-------------------|----------------------|--------------------|---------------------|-------------------------|------------------|-------|
| 1 |                   |                      |                    |                     |                         |                  |       |
| 2 |                   |                      |                    |                     |                         |                  |       |
| … |                   |                      |                    |                     |                         |                  |       |

Suma / 40. Porównuj:

| Model | Wariant promptu | Wynik /40 | Czas odp. | Uwagi |
|-------|-----------------|-----------|-----------|-------|
| Qwen2.5-3B Q4 | zero-shot (sam SYSTEM) | | | |
| Qwen2.5-3B Q4 | 6 przykładów | | | |
| Qwen2.5-3B Q4 | 20 przykładów | | | |
| Llama-3.2-3B Q4 | 6 przykładów | | | |
| Qwen2.5-7B Q4 (tylko 16 GB) | 6 przykładów | | | |

Decyzja po arkuszu:
- **≥ 32/40 z 6–20 przykładami** → few-shot wystarcza; fine-tuning nie jest potrzebny. Buduj narzędzie wokół promptu.
- **20–31** → model "prawie umie". Fine-tuning (LoRA na Macu, lekcja 9) na 200–500 przykładach prawdopodobnie domknie lukę.
- **< 20** → problem jest w danych/definicji zadania, nie w modelu. Wróć do przykładów: czy kategorie są rozłączne? czy człowiek z tymi samymi metrykami odpowiedziałby jednoznacznie?
