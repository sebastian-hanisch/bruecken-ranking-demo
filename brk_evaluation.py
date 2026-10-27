"""Auswertung: Einstellungen, drei Rangverfahren gegen die wahre Qualität, Tabellen, Verdikte als reine Funktionen (ohne Streamlit) und die festen Reihen."""

import math
from dataclasses import dataclass

import numpy as np

import brk_algorithm as A
import brk_constants as C
import brk_scenario as S


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f} %".replace(".", ",").replace("-", "−")


@dataclass(frozen=True)
class Settings:
    users: int = C.DEFAULT_U
    notes: int = C.DEFAULT_N
    share: int = C.DEFAULT_SHARE
    gamma: float = C.DEFAULT_GAMMA
    partisan: int = C.DEFAULT_PARTISAN
    obs: int = C.DEFAULT_OBS
    lam_i: float = C.DEFAULT_LAMI
    lam_f: float = C.DEFAULT_LAMF
    seed: int = C.DEFAULT_SEED
    text: str = ""


@dataclass
class Analysis:
    settings: Settings
    world: S.World
    source: str              # "eingegeben" oder "erzeugt"
    errors: list
    users: int
    notes: int
    scores: dict             # Verfahren -> Wert je Vorschlag
    tops: dict               # Verfahren -> Indizes der besten Vorschläge
    metrics: dict            # Verfahren -> {"spearman", "partisan", "quality", "lean_a"} (None bei eingegebenen Bewertungen)
    fit: dict                # Ergebnis der Faktorisierung
    recovery: float          # Betrag der Korrelation des Bewerterfaktors mit dem Lager
    count: int               # so viele Vorschläge zählen als „oben“


def analyse(settings: Settings) -> Analysis:
    errors = []
    world, err = S.parse_ratings(settings.text)
    if err:
        errors.append("Eigene Bewertungen: " + err + " Die Eingabe wird ignoriert.")
    if world is not None:
        source = "eingegeben"
    else:
        world = S.make_world(settings.users, settings.notes, settings.share, settings.gamma, settings.partisan, settings.obs, settings.seed)
        source = "erzeugt"
    users, notes = world.ratings.shape
    fit = A.fit_bridging(world.ratings, world.mask, settings.lam_i, settings.lam_f, seed=settings.seed + 2)
    scores = {"mittel": A.mean_scores(world.ratings, world.mask), "lager": A.min_camp_scores(world.ratings, world.mask, world.camp), "mf": fit["inn"]}
    count = min(C.TOPK, notes)
    tops = {k: A.top_k(v, count) for k, v in scores.items()}
    metrics = None
    if world.quality is not None:
        metrics = {k: {"spearman": A.spearman(scores[k], world.quality), "partisan": float(np.mean(world.lean[tops[k]] != 0)), "quality": float(world.quality[tops[k]].mean()),
                       "lean_a": float(np.mean(world.lean[tops[k]] == 1))} for k in scores}
    return Analysis(settings=settings, world=world, source=source, errors=errors, users=users, notes=notes, scores=scores, tops=tops, metrics=metrics, fit=fit,
                    recovery=A.camp_recovery(fit["fu"], world.camp), count=count)


# --- Tabellen und Texte ---------------------------------------------------------------------------------------------------------------------

def summary_rows(a: Analysis) -> list:
    if a.metrics is None:
        return []
    return [{"Verfahren": C.RANKERS[k], "Rangkorrelation zur Qualität": num(a.metrics[k]["spearman"], 2), f"Parteiisch unter den besten {a.count}": pct(a.metrics[k]["partisan"]),
             f"Neigung zu Lager A unter den besten {a.count}": pct(a.metrics[k]["lean_a"]), f"Mittlere Qualität der besten {a.count}": num(a.metrics[k]["quality"], 2)} for k in ("mittel", "lager", "mf")]


def ranking_rows(a: Analysis, limit: int = 10) -> list:
    """Die besten Vorschläge je Verfahren nebeneinander (mit Qualität und Neigung, wenn bekannt)."""
    rows = []
    for pos in range(min(limit, a.count)):
        row = {"Platz": pos + 1}
        for k in ("mittel", "lager", "mf"):
            j = int(a.tops[k][pos])
            row[C.RANKER_SHORT[k]] = f"Vorschlag {j + 1}" + (f" ({lean_word(int(a.world.lean[j]))}, Qualität {num(float(a.world.quality[j]), 1)})" if a.world.quality is not None else "")
        rows.append(row)
    return rows


def lean_word(lean: int) -> str:
    return {1: "Lager A", -1: "Lager B", 0: "neutral"}[lean]


def score_rows(a: Analysis) -> list:
    """Werte je Vorschlag und Verfahren (für eigene Bewertungen)."""
    return [{"Vorschlag": j + 1, "Mittlere Zustimmung": num(float(a.scores["mittel"][j]), 3), "Mindestzustimmung": num(float(a.scores["lager"][j]), 3), "Intercept (brückenbasiert)": num(float(a.scores["mf"][j]), 3)} for j in range(a.notes)]


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------

def verdict_state(a: Analysis):
    """Kernfrage: wählt das brückenbasierte Verfahren die besseren Vorschläge als der Mittelwert? Rückgabe (st-Methode, Text)."""
    if a.metrics is None:
        top = {k: int(a.tops[k][0]) + 1 for k in a.tops}
        tied = int(np.sum(np.isclose(a.scores["mittel"], a.scores["mittel"].max())))
        text = (f"Auf Platz 1 stehen: Mittelwert Vorschlag {top['mittel']}, Mindestzustimmung Vorschlag {top['lager']}, brückenbasiert Vorschlag {top['mf']}."
                + (f" Im Mittelwert teilen sich {tied} Vorschläge den ersten Platz; die Mindestzustimmung unterscheidet sie nach der Zustimmung des schwächeren Lagers." if tied > 1 else ""))
        return "info", text
    m, b = a.metrics["mittel"], a.metrics["mf"]
    d_rho = b["spearman"] - m["spearman"]
    head = (f"Rangkorrelation zur Qualität: Mittelwert {num(m['spearman'], 2)}, brückenbasiert {num(b['spearman'], 2)}. Unter den besten {a.count} sind {pct(m['partisan'])} (Mittelwert) und {pct(b['partisan'])} (brückenbasiert) parteiische Vorschläge; "
            f"die Lagerrichtung steckt zu {pct(a.recovery)} im Bewerterfaktor.")
    if not a.fit["converged"]:
        head += f" Die Faktorisierung hat nach {a.fit['steps']} Schritten die Toleranz nicht erreicht."
    if d_rho > 0.02 and b["partisan"] < m["partisan"]:
        return "success", head
    if d_rho < -0.02:
        return "warning", head + " Hier ist das Verfahren schlechter als der Mittelwert (Regularisierung prüfen)."
    return "info", head + " Beide Verfahren liegen nahe beieinander."


def verdict_share(rows):
    """Reihe 1: wirkt die Brücke erst bei ungleich großen Lagern? Rückgabe (st-Methode, Text)."""
    first, last = rows[0], rows[-1]
    text = (f"Bei gleich großen Lagern ({first['share']} : {100 - first['share']}) liegen Mittelwert und brückenbasiertes Verfahren bei {num(first['mittel']['spearman'], 2)} und {num(first['mf']['spearman'], 2)} (Rangkorrelation zur Qualität); "
            f"bei {last['share']} : {100 - last['share']} fällt der Mittelwert auf {num(last['mittel']['spearman'], 2)}, das brückenbasierte Verfahren auf {num(last['mf']['spearman'], 2)}. "
            f"Parteiische Vorschläge unter den besten {C.TOPK}: Mittelwert {pct(first['mittel']['partisan'])} auf {pct(last['mittel']['partisan'])}, brückenbasiert {pct(first['mf']['partisan'])} auf {pct(last['mf']['partisan'])}.")
    return "info", text


def verdict_lam(rows):
    """Reihe 2: wie wichtig ist die Regularisierung? Rückgabe (st-Methode, Text)."""
    weak, strong = rows[0], rows[-1]
    ref = weak["mittel"]["spearman"]
    text = (f"Mit schwacher Regularisierung des Intercepts (λ = {num(weak['lam_i'], 2)}) liegt das brückenbasierte Verfahren bei {num(weak['mf']['spearman'], 2)} und {pct(weak['mf']['partisan'])} parteiischen Vorschlägen unter den besten {C.TOPK}, "
            f"der Mittelwert bei {num(ref, 2)} und {pct(weak['mittel']['partisan'])}; mit λ = {num(strong['lam_i'], 1)} sind es {num(strong['mf']['spearman'], 2)} und {pct(strong['mf']['partisan'])}.")
    return ("success" if strong["mf"]["spearman"] > weak["mf"]["spearman"] + 0.05 else "info"), text


def verdict_gamma(rows):
    """Reihe 3: was ändert die Stärke der Lagerneigung? Rückgabe (st-Methode, Text)."""
    first, last = rows[0], rows[-1]
    text = (f"Ohne Lagerneigung sind alle drei Verfahren gleich gut (Rangkorrelation {num(first['mittel']['spearman'], 2)} / {num(first['lager']['spearman'], 2)} / {num(first['mf']['spearman'], 2)}); "
            f"bei Neigung {num(last['gamma'], 1)} sind es {num(last['mittel']['spearman'], 2)} / {num(last['lager']['spearman'], 2)} / {num(last['mf']['spearman'], 2)} für Mittelwert, Mindestzustimmung und brückenbasiert.")
    return "info", text


# --- Feste Reihen ---------------------------------------------------------------------------------------------------------------------------

def _run(share, gamma, lam_i, seed, lam_f=C.DEFAULT_LAMF):
    return analyse(Settings(share=share, gamma=gamma, lam_i=lam_i, lam_f=lam_f, seed=seed))


def _aggregate(analyses) -> dict:
    out = {k: {m: float(np.mean([a.metrics[k][m] for a in analyses])) for m in ("spearman", "partisan", "quality", "lean_a")} for k in ("mittel", "lager", "mf")}
    out["recovery"] = float(np.mean([a.recovery for a in analyses]))
    out["converged"] = sum(1 for a in analyses if a.fit["converged"])
    out["runs"] = len(analyses)
    return out


def series_share() -> list:
    """Je Größe des großen Lagers: Maße der drei Verfahren (dieselben Welten bis auf die Lagerzuteilung)."""
    return [{"share": share, **_aggregate([_run(share, C.DEFAULT_GAMMA, C.DEFAULT_LAMI, C.SERIES_SEED + run) for run in range(C.SERIES_RUNS)])} for share in C.SERIES_SHARES]


def series_lam() -> list:
    """Je Regularisierung des Intercepts bei 80 : 20 (dieselben Welten in jeder Zeile)."""
    return [{"lam_i": lam, **_aggregate([_run(C.SERIES_SHARE_MAIN, C.DEFAULT_GAMMA, lam, C.SERIES_SEED + 1000 + run) for run in range(C.SERIES_RUNS)])} for lam in C.SERIES_LAMIS]


def series_gamma() -> list:
    """Je Stärke der Lagerneigung bei 80 : 20 (dieselben Welten in jeder Zeile)."""
    return [{"gamma": g, **_aggregate([_run(C.SERIES_SHARE_MAIN, g, C.DEFAULT_LAMI, C.SERIES_SEED + 2000 + run) for run in range(C.SERIES_RUNS)])} for g in C.SERIES_GAMMAS]


def _cells(r) -> dict:
    return {"Mittelwert": num(r["mittel"]["spearman"], 2), "Mindestlager": num(r["lager"]["spearman"], 2), "Brückenbasiert": num(r["mf"]["spearman"], 2),
            "Parteiisch oben (Mittelwert)": pct(r["mittel"]["partisan"]), "Parteiisch oben (Mindestlager)": pct(r["lager"]["partisan"]), "Parteiisch oben (brückenbasiert)": pct(r["mf"]["partisan"])}


def share_table(rows) -> list:
    return [{"Lager": f"{r['share']} : {100 - r['share']}", **_cells(r), "Lager im Faktor": pct(r["recovery"])} for r in rows]


def lam_table(rows) -> list:
    return [{"λ Intercept": num(r["lam_i"], 2), **_cells(r), "Konvergiert": f"{r['converged']} von {r['runs']}"} for r in rows]


def gamma_table(rows) -> list:
    return [{"Lagerneigung": num(r["gamma"], 1), **_cells(r), "Lager im Faktor": pct(r["recovery"])} for r in rows]
