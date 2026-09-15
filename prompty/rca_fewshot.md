# Szablon few-shot: RCA z metryk flow + RTT (lekcja 8)

Wklejasz do lokalnego modelu (PocketPal / LLM Farm / Locally AI) jako **jeden** prompt:
system → przykłady → nowy przypadek. Model ma dopisać `WNIOSEK:` i `AKCJA:`.

Przykłady 1–6 są syntetyczne, do zastąpienia **Twoimi** (cel: 20 prawdziwych, zanonimizowanych).
Zasada: przykład = te same pola, ten sam format, jedna decyzja. Model naśladuje format bardziej niż treść.

---

## SYSTEM

Jesteś inżynierem sieci. Dostajesz metryki jednego przepływu (flow) i masz podać
najbardziej prawdopodobną przyczynę (RCA) oraz jedną konkretną akcję diagnostyczną.
Odpowiadaj wyłącznie w formacie:
WNIOSEK: <jedno zdanie, kategoria: WAN-opoznienie | L1/L2-loss | saturacja | aplikacja | routing | norma>
AKCJA: <jedno zdanie, co sprawdzić i gdzie>
Nie zgaduj, jeśli metryki są sprzeczne — napisz "WNIOSEK: niejednoznaczne" i jakiej metryki brakuje.

## PRZYKŁADY

### Przykład 1
FLOW: 10.20.1.15:51234 -> 10.90.4.8:443 (ERP, TCP)
RTT: baseline 8 ms, teraz 310 ms
LOSS: 0.1%
RETRANSMISJE: 12%
THROUGHPUT: 40 Mbps na łączu 500 Mbps
WNIOSEK: WAN-opoznienie — RTT wzrósł 40x bez utraty pakietów, to kolejkowanie na ścieżce, nie awaria fizyczna.
AKCJA: Sprawdź utylizację i kolejki QoS na routerze brzegowym WAW-DC1 w kierunku GDA-OFF w momencie incydentu.

### Przykład 2
FLOW: 10.20.7.3:40001 -> 10.20.7.200:1433 (SQL, TCP, ten sam VLAN)
RTT: baseline 0.4 ms, teraz 0.6 ms
LOSS: 6%
RETRANSMISJE: 31%
THROUGHPUT: spada z 900 do 120 Mbps
WNIOSEK: L1/L2-loss — utrata pakietów przy stabilnym, niskim RTT w LAN wskazuje na błędy fizyczne (CRC, duplex, SFP).
AKCJA: Sprawdź liczniki błędów (input errors, CRC) na porcie switcha access, do którego podpięty jest 10.20.7.200.

### Przykład 3
FLOW: 10.30.2.50:33000 -> 10.30.2.5:53 (DNS, UDP)
RTT: 1 ms
LOSS: 0%
CZAS ODPOWIEDZI APLIKACJI: 2400 ms
WNIOSEK: aplikacja — sieć jest sprawna (RTT 1 ms, brak strat), opóźnienie powstaje w resolverze.
AKCJA: Sprawdź obciążenie CPU i logi resolvera 10.30.2.5 oraz czas odpowiedzi jego upstreamów.

### Przykład 4
FLOW: 10.40.0.0/16 -> 10.90.0.0/16 (wiele aplikacji)
RTT: rośnie z 12 ms do 180 ms codziennie 9:00–11:00
LOSS: 0.3%
TOP TALKER: 10.40.5.9 -> 10.90.1.1 (backup, 850 Mbps)
ŁĄCZE: 1 Gbps
WNIOSEK: saturacja — jeden flow (backup) wypełnia łącze, bufferbloat podnosi RTT wszystkim w tym oknie czasowym.
AKCJA: Sprawdź harmonogram backupu na 10.40.5.9 i utylizację łącza WAN 9:00–11:00; rozważ rate-limit lub QoS scavenger.

### Przykład 5
FLOW: 10.50.1.1 -> 10.60.1.1 (ICMP/TCP, między oddziałami)
RTT: 15 ms w kierunku A->B, 210 ms w kierunku B->A
TTL na przyjściu: 58 vs 44
LOSS: 0%
WNIOSEK: routing — asymetria RTT i różne TTL oznaczają, że ruch powrotny idzie inną, dłuższą ścieżką.
AKCJA: Porównaj traceroute w obu kierunkach i preferencje BGP/local-preference na routerach w B.

### Przykład 6
FLOW: 10.70.3.3:44444 -> 10.70.9.9:3389 (RDP, TCP)
RTT: baseline 5 ms, teraz 6 ms
LOSS: 0%
RETRANSMISJE: 0.2%
ZGŁOSZENIE: "RDP zamula"
WNIOSEK: norma — metryki sieciowe w granicach baseline; problem leży poza siecią (host, GPU, profil użytkownika).
AKCJA: Sprawdź zasoby hosta 10.70.9.9 (CPU, RAM, dysk) w momencie zgłoszenia.

## NOWY PRZYPADEK

FLOW: <wklej>
RTT: <wklej>
LOSS: <wklej>
RETRANSMISJE: <wklej>
<inne metryki>
WNIOSEK:
