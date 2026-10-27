"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout. Die Reihen mitteln über Zufallswelten mit festen Zufalls-Seeds (in jeder Reihe dieselben Welten je Zeile)."""

import sys
import time

sys.path.insert(0, ".")
import brk_constants as C
import brk_evaluation as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    a = ev.analyse(ev.Settings(**p))
    print(name, a.users, a.notes, "Schritte", a.fit["steps"], a.fit["converged"])
    print("   ", ev.verdict_state(a))
    for r in ev.summary_rows(a) or ev.score_rows(a):
        print("   ", r)

for title, fn, table, verdict in (("REIHE 1 (Größe des großen Lagers)", ev.series_share, ev.share_table, ev.verdict_share),
                                  ("REIHE 2 (Regularisierung des Intercepts)", ev.series_lam, ev.lam_table, ev.verdict_lam),
                                  ("REIHE 3 (Lagerneigung)", ev.series_gamma, ev.gamma_table, ev.verdict_gamma)):
    t = time.time()
    rows = fn()
    print(title, round(time.time() - t, 1), "s")
    for r in table(rows):
        print("   ", r)
    print("   ", verdict(rows))
print("sec", round(time.time() - t0, 1))
