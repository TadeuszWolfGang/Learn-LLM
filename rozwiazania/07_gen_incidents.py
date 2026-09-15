"""
LEKCJA 7 - generator zabawkowego korpusu "incydentów sieciowych".

Mikro-model znakowy potrzebuje >= ~200 KB tekstu, żeby coś pokazać.  Ręcznie tyle
nie napiszesz, więc generujemy z szablonów.  Sens: zobaczyć, JAKI rodzaj struktury
model wyłapie (format, słownictwo, korelacje "RTT wysoki -> retransmisje"),
a czego NIE (przyczynowość, liczby jako liczby).

    python3 07_gen_incidents.py            # -> data/incidents.txt (~250 KB)
    python3 07_gen_incidents.py --n 500    # mniej wpisów

Potem: python3 05_train_gpt.py --data ../data/incidents.txt --out out_inc
"""
import argparse, os, random

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1500)
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "incidents.txt"))
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
rng = random.Random(a.seed)

sites = ["WAW-DC1", "WAW-DC2", "KRK-OFF", "GDA-OFF", "POZ-WH", "AWS-EU1", "AZR-WE"]
apps = ["ERP", "CRM", "VoIP", "backup", "VDI", "S3-sync", "DNS", "LDAP", "Jira", "SAP"]
protos = ["TCP/443", "TCP/1433", "UDP/5060", "TCP/22", "TCP/3389", "UDP/53", "TCP/8443"]

def ip():
    return f"10.{rng.randint(0,250)}.{rng.randint(0,250)}.{rng.randint(1,250)}"

def incident():
    kind = rng.choice(["rtt", "loss", "bw", "dns", "asym", "ok"])
    src, dst = rng.choice(sites), rng.choice(sites)
    app, proto = rng.choice(apps), rng.choice(protos)
    hh, mm = rng.randint(0, 23), rng.randint(0, 59)
    lines = [f"[{hh:02d}:{mm:02d}] INCYDENT {rng.randint(1000,9999)} | {src} -> {dst} | {app} {proto}",
             f"flow: {ip()}:{rng.randint(1024,65000)} -> {ip()}:{proto.split('/')[1]}"]
    if kind == "rtt":
        base, now = rng.randint(3, 30), rng.randint(120, 900)
        lines += [f"RTT: baseline {base} ms, teraz {now} ms (x{now//base})",
                  f"retransmisje: {rng.randint(3,25)}% (norma <1%)",
                  f"loss: {rng.choice(['0','0.1','0.5'])}%",
                  "wniosek: opoznienie na sciezce WAN, bez utraty pakietow -> saturacja lacza lub kolejkowanie QoS",
                  f"akcja: sprawdz utylizacje {rng.choice(['MPLS','SD-WAN','VPN'])} {src}, QoS klasa {app}"]
    elif kind == "loss":
        lines += [f"RTT: baseline {rng.randint(3,30)} ms, teraz {rng.randint(5,60)} ms (stabilne)",
                  f"loss: {rng.randint(2,15)}% (norma 0%)",
                  f"retransmisje: {rng.randint(5,40)}%",
                  "wniosek: utrata pakietow przy niskim RTT -> problem L1/L2 (CRC, duplex, wadliwy SFP)",
                  f"akcja: sprawdz bledy interfejsu na switchu {rng.choice(['core','dist','acc'])}-{rng.randint(1,8)} w {dst}"]
    elif kind == "bw":
        lines += [f"throughput: {rng.randint(400,950)} Mbps na laczu {rng.choice([500,1000])} Mbps",
                  f"RTT: rosnie z {rng.randint(3,20)} ms do {rng.randint(40,200)} ms w godzinach {rng.randint(8,11)}-{rng.randint(12,17)}",
                  f"top talker: {ip()} ({rng.choice(['backup','S3-sync','VDI','Windows Update'])})",
                  "wniosek: saturacja lacza przez pojedynczy flow -> RTT rosnie dla wszystkich (bufferbloat)",
                  "akcja: rate-limit lub przeniesienie zadania poza godziny szczytu"]
    elif kind == "dns":
        lines += [f"czas odpowiedzi DNS: {rng.randint(800,5000)} ms (norma <20 ms)",
                  f"RTT do resolvera: {rng.randint(1,5)} ms",
                  f"loss: 0%",
                  "wniosek: siec sprawna (niski RTT, brak loss), wolna jest aplikacja/resolver -> nie problem sieciowy",
                  f"akcja: eskalacja do zespolu {rng.choice(['DNS','Windows','Linux'])}, sprawdz obciazenie resolvera"]
    elif kind == "asym":
        lines += [f"RTT: {rng.randint(3,20)} ms w jedna strone, {rng.randint(100,400)} ms w druga",
                  f"TTL: rozne wartosci na sciezkach tam/powrot ({rng.randint(50,60)} vs {rng.randint(40,49)})",
                  "wniosek: routing asymetryczny -> powrot idzie inna, dluzsza sciezka",
                  f"akcja: sprawdz tablice routingu i BGP preferencje w {src}"]
    else:
        lines += [f"RTT: {rng.randint(2,25)} ms (baseline)", "loss: 0%", "retransmisje: <1%",
                  "wniosek: parametry sieci w normie, zgloszenie uzytkownika nie potwierdzone",
                  "akcja: zamkniete, monitoring 24h"]
    return "\n".join(lines) + "\n\n"

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for _ in range(a.n):
        f.write(incident())
print(f"zapisano {a.out}: {os.path.getsize(a.out)/1024:.0f} KB, {a.n} incydentow")
