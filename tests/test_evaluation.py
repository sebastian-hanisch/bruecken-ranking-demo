"""Auswertung: Formatierung, Analyse, Tabellen, Verdikte und die Zahlen in den Hilfetexten der Presets."""

from types import SimpleNamespace

import numpy as np
import pytest

import brk_constants as C
import brk_evaluation as ev
from brk_evaluation import Settings, analyse

HAND = C.PRESETS["Handbeispiel (sechs Bewerter)"]["text"]


def test_formatierung():
    assert ev.num(0.936, 2) == "0,94" and ev.num(-0.5, 1) == "−0,5"
    assert ev.pct(0.296) == "30 %" and ev.pct(0.296, 1) == "29,6 %" and ev.pct(-0.1) == "−10 %"
    assert ev.pct(None) == "–" and ev.pct(float("nan")) == "–"
    assert ev.lean_word(1) == "Lager A" and ev.lean_word(-1) == "Lager B" and ev.lean_word(0) == "neutral"


def test_analyse_ist_deterministisch_und_konsistent():
    a, b = analyse(Settings()), analyse(Settings())
    assert np.array_equal(a.scores["mf"], b.scores["mf"]) and a.metrics == b.metrics
    assert (a.users, a.notes, a.source, a.errors, a.count) == (C.DEFAULT_U, C.DEFAULT_N, "erzeugt", [], C.TOPK)
    assert set(a.scores) == set(a.tops) == set(a.metrics) == {"mittel", "lager", "mf"} and all(len(t) == a.count for t in a.tops.values())
    for k, m in a.metrics.items():
        assert -1 <= m["spearman"] <= 1 and 0 <= m["partisan"] <= 1 and 0 <= m["lean_a"] <= m["partisan"] + 1e-12
        assert m["partisan"] == pytest.approx(np.mean(a.world.lean[a.tops[k]] != 0)) and m["quality"] == pytest.approx(a.world.quality[a.tops[k]].mean())
    assert 0 <= a.recovery <= 1 and a.fit["converged"]


def test_zahl_der_besten_ist_höchstens_die_zahl_der_vorschläge():
    a = analyse(Settings(notes=C.N_MIN))
    assert a.count == C.TOPK == 10
    b = analyse(Settings(text="A 1 0\nB 0 1\nA 1 1"))
    assert b.count == 2 and len(b.tops["mf"]) == 2


def test_handbeispiel_exakte_werte():
    a = analyse(Settings(text=HAND))
    assert (a.source, a.users, a.notes, a.metrics) == ("eingegeben", 6, 4, None)
    assert a.scores["mittel"] == pytest.approx([5 / 6, 5 / 6, 1 / 6, 1 / 3])
    assert a.scores["lager"] == pytest.approx([0.5, 0.75, 0.0, 0.0])
    assert [int(a.tops[k][0]) for k in ("mittel", "lager", "mf")] == [0, 1, 0]                 # Gleichstand im Mittel: kleinerer Index; Mindestlager: Vorschlag 2
    assert a.scores["mf"] == pytest.approx([0.186, 0.071, -0.109, -0.166], abs=2e-3)
    assert ev.summary_rows(a) == []


def test_ungültige_eingabe_wird_gemeldet_und_ignoriert():
    a = analyse(Settings(users=50, notes=12, text="A 1 0\nC 0 1\nA 1 1"))
    assert a.source == "erzeugt" and (a.users, a.notes) == (50, 12)
    assert len(a.errors) == 1 and a.errors[0].startswith("Eigene Bewertungen:") and "„C“" in a.errors[0]


# --- Tabellen ---------------------------------------------------------------------------------------------------------------------------------


def test_tabellen_der_analyse():
    a = analyse(Settings())
    rows = ev.summary_rows(a)
    assert [r["Verfahren"] for r in rows] == [C.RANKERS[k] for k in ("mittel", "lager", "mf")]
    assert rows[0]["Rangkorrelation zur Qualität"] == ev.num(a.metrics["mittel"]["spearman"], 2)
    assert rows[2][f"Parteiisch unter den besten {a.count}"] == ev.pct(a.metrics["mf"]["partisan"])
    ranking = ev.ranking_rows(a)
    assert len(ranking) == a.count and ranking[0]["Platz"] == 1
    j = int(a.tops["mf"][0])
    assert ranking[0]["Brückenbasiert"] == f"Vorschlag {j + 1} ({ev.lean_word(int(a.world.lean[j]))}, Qualität {ev.num(float(a.world.quality[j]), 1)})"


def test_wertetabelle_für_eigene_bewertungen():
    a = analyse(Settings(text=HAND))
    rows = ev.score_rows(a)
    assert [r["Vorschlag"] for r in rows] == [1, 2, 3, 4]
    assert rows[1]["Mittlere Zustimmung"] == "0,833" and rows[1]["Mindestzustimmung"] == "0,750" and rows[2]["Mindestzustimmung"] == "0,000"
    assert ev.ranking_rows(a)[0] == {"Platz": 1, "Mittelwert": "Vorschlag 1", "Mindestlager": "Vorschlag 2", "Brückenbasiert": "Vorschlag 1"}


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------


def _fake(m_rho, b_rho, m_part, b_part, converged=True, steps=50):
    metrics = {"mittel": {"spearman": m_rho, "partisan": m_part, "quality": 1.0, "lean_a": 0.0}, "mf": {"spearman": b_rho, "partisan": b_part, "quality": 1.0, "lean_a": 0.0}}
    return SimpleNamespace(metrics=metrics, count=10, recovery=0.97, fit={"converged": converged, "steps": steps})


def test_verdikt_erfolg_warnung_info():
    kind, text = ev.verdict_state(_fake(0.75, 0.94, 0.5, 0.3))
    assert kind == "success" and "Mittelwert 0,75, brückenbasiert 0,94" in text and "50 % (Mittelwert) und 30 % (brückenbasiert)" in text and "zu 97 % im Bewerterfaktor" in text
    kind, text = ev.verdict_state(_fake(0.90, 0.80, 0.3, 0.3))
    assert kind == "warning" and "schlechter als der Mittelwert" in text
    kind, text = ev.verdict_state(_fake(0.93, 0.95, 0.4, 0.4))
    assert kind == "info" and "nahe beieinander" in text


def test_verdikt_schwellen_der_unterschiede():
    assert ev.verdict_state(_fake(0.75, 0.7699, 0.5, 0.3))[0] == "info"                         # Unterschied unter 0,02 reicht nicht
    assert ev.verdict_state(_fake(0.75, 0.7701, 0.5, 0.3))[0] == "success"
    assert ev.verdict_state(_fake(0.75, 0.94, 0.5, 0.5))[0] == "info"                         # gleich viele parteiische: kein Erfolg
    assert ev.verdict_state(_fake(0.75, 0.7301, 0.5, 0.3))[0] == "info" and ev.verdict_state(_fake(0.75, 0.7299, 0.5, 0.3))[0] == "warning"


def test_verdikt_nennt_nichtkonvergenz():
    kind, text = ev.verdict_state(_fake(0.75, 0.94, 0.5, 0.3, converged=False, steps=300))
    assert "nach 300 Schritten die Toleranz nicht erreicht" in text


def test_verdikt_eigene_bewertungen_nennt_gleichstand_der_spitze():
    kind, text = ev.verdict_state(analyse(Settings(text=HAND)))
    assert kind == "info" and "Mittelwert Vorschlag 1, Mindestzustimmung Vorschlag 2, brückenbasiert Vorschlag 1" in text and "teilen sich 2 Vorschläge den ersten Platz" in text
    kind, text = ev.verdict_state(analyse(Settings(text="A 1 0\nB 1 0\nA 1 0")))
    assert "teilen sich" not in text


def _row(**kw):
    base = {"runs": 30, "recovery": 0.97, "converged": 30}
    for k in ("mittel", "lager", "mf"):
        base[k] = {"spearman": 0.8, "partisan": 0.5, "quality": 1.5, "lean_a": 0.2}
    base.update(kw)
    return base


def _m(sp, part):
    return {"spearman": sp, "partisan": part, "quality": 1.5, "lean_a": 0.2}


def test_verdikt_reihe_lagergröße():
    rows = [_row(share=50, mittel=_m(0.94, 0.34), mf=_m(0.95, 0.33)), _row(share=90, mittel=_m(0.69, 0.76), mf=_m(0.92, 0.37))]
    kind, text = ev.verdict_share(rows)
    assert kind == "info" and "(50 : 50) liegen Mittelwert und brückenbasiertes Verfahren bei 0,94 und 0,95" in text and "bei 90 : 10 fällt der Mittelwert auf 0,69, das brückenbasierte Verfahren auf 0,92" in text
    assert "Mittelwert 34 % auf 76 %, brückenbasiert 33 % auf 37 %" in text


def test_verdikt_reihe_regularisierung():
    rows = [_row(lam_i=0.15, mittel=_m(0.77, 0.57), mf=_m(0.80, 0.50)), _row(lam_i=10.0, mittel=_m(0.77, 0.57), mf=_m(0.94, 0.28))]
    kind, text = ev.verdict_lam(rows)
    assert kind == "success" and "(λ = 0,15) liegt das brückenbasierte Verfahren bei 0,80 und 50 %" in text and "der Mittelwert bei 0,77 und 57 %" in text and "mit λ = 10,0 sind es 0,94 und 28 %" in text
    assert ev.verdict_lam([_row(lam_i=0.15, mf=_m(0.80, 0.5)), _row(lam_i=10.0, mf=_m(0.8501, 0.5))])[0] == "success"            # Zuwachs über 0,05
    assert ev.verdict_lam([_row(lam_i=0.15, mf=_m(0.80, 0.5)), _row(lam_i=10.0, mf=_m(0.8499, 0.5))])[0] == "info"


def test_verdikt_reihe_lagerneigung():
    rows = [_row(gamma=0.0, mittel=_m(0.99, 0.6), lager=_m(0.97, 0.6), mf=_m(0.98, 0.6)), _row(gamma=3.0, mittel=_m(0.50, 0.55), lager=_m(0.42, 0.0), mf=_m(0.69, 0.04))]
    kind, text = ev.verdict_gamma(rows)
    assert kind == "info" and "0,99 / 0,97 / 0,98" in text and "bei Neigung 3,0 sind es 0,50 / 0,42 / 0,69" in text


def test_reihentabellen():
    r = _row(share=80, mittel=_m(0.76, 0.59), lager=_m(0.77, 0.17), mf=_m(0.94, 0.36))
    cells = ev.share_table([r])[0]
    assert cells["Lager"] == "80 : 20" and cells["Mittelwert"] == "0,76" and cells["Brückenbasiert"] == "0,94" and cells["Parteiisch oben (Mindestlager)"] == "17 %" and cells["Lager im Faktor"] == "97 %"
    assert ev.lam_table([_row(lam_i=0.15)])[0]["λ Intercept"] == "0,15" and ev.lam_table([_row(lam_i=0.15)])[0]["Konvergiert"] == "30 von 30"
    assert ev.gamma_table([_row(gamma=1.5)])[0]["Lagerneigung"] == "1,5"


# --- Presets: jede Zahl im Hilfetext stimmt ---------------------------------------------------------------------------------------------------

EXPECTED = {
    "Ungleiche Lager (80 : 20)": {"mittel": "0,75", "lager": "0,74", "mf": "0,94", "part_mittel": "50 %", "part_mf": "30 %"},
    "Gleich große Lager": {"mittel": "0,93", "lager": "0,77", "mf": "0,95"},
    "Schwache Regularisierung": {"mittel": "0,75", "mf": "0,78", "part_mf": "50 %"},
    "Keine Lagerneigung": {"mittel": "0,98", "lager": "0,97", "mf": "0,98"},
    "Starke Polarisierung": {"mittel": "0,51", "lager": "0,35", "mf": "0,70", "part_mittel": "50 %", "part_mf": "0 %"},
}


def test_jedes_preset_hat_einen_hilfetext():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and all(C.PRESET_HELP.values()) and set(EXPECTED) | {"Handbeispiel (sechs Bewerter)"} == set(C.PRESETS)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_hilfetext_des_presets_stimmt_mit_der_rechnung(name):
    a = analyse(Settings(**C.PRESETS[name]))
    exp, help_text = EXPECTED[name], C.PRESET_HELP[name]
    for key in ("mittel", "lager", "mf"):
        if key in exp:
            assert ev.num(a.metrics[key]["spearman"], 2) == exp[key] and exp[key] in help_text
    if "part_mittel" in exp:
        assert ev.pct(a.metrics["mittel"]["partisan"]) == exp["part_mittel"] and exp["part_mittel"] in help_text
    if "part_mf" in exp:
        assert ev.pct(a.metrics["mf"]["partisan"]) == exp["part_mf"] and exp["part_mf"] in help_text
    assert a.fit["converged"]


def test_hilfetext_handbeispiel_nennt_die_brüche():
    text = C.PRESET_HELP["Handbeispiel (sechs Bewerter)"]
    assert "5/6" in text and "1/2 gegen 3/4" in text and "Vorschlag 2 vorn" in text


def test_preset_aussagen_untereinander():
    a = analyse(Settings(**C.PRESETS["Ungleiche Lager (80 : 20)"]))
    weak = analyse(Settings(**C.PRESETS["Schwache Regularisierung"]))
    assert np.array_equal(a.world.ratings, weak.world.ratings) and a.metrics["mittel"] == weak.metrics["mittel"]       # nur die Regularisierung unterscheidet beide
    assert a.metrics["mf"]["spearman"] > weak.metrics["mf"]["spearman"] + 0.1
    none = analyse(Settings(**C.PRESETS["Keine Lagerneigung"]))
    assert none.recovery < 0.1 and a.recovery > 0.9
    strong = analyse(Settings(**C.PRESETS["Starke Polarisierung"]))
    assert strong.metrics["mf"]["partisan"] == 0.0 < strong.metrics["mittel"]["partisan"]


# --- Reihen mit ersetztem Lauf ----------------------------------------------------------------------------------------------------------------


def _fake_run(calls):
    def fake(*args, **kwargs):
        calls.append(args)
        metrics = {k: {"spearman": 0.5 + 0.1 * i, "partisan": 0.2 + 0.1 * i, "quality": 1.0, "lean_a": 0.0} for i, k in enumerate(("mittel", "lager", "mf"))}
        return SimpleNamespace(metrics=metrics, recovery=0.9, fit={"converged": True})
    return fake


def test_reihen_rechnen_alle_läufe_mit_gekoppelten_seeds_und_aggregieren(monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "_run", _fake_run(calls))
    rows = ev.series_share()
    assert len(calls) == len(C.SERIES_SHARES) * C.SERIES_RUNS and {c[3] for c in calls} == {C.SERIES_SEED + r for r in range(C.SERIES_RUNS)}      # dieselben Welten in jeder Zeile
    assert [r["share"] for r in rows] == list(C.SERIES_SHARES) and {c[0] for c in calls} == set(C.SERIES_SHARES)
    assert all(r["mittel"]["spearman"] == pytest.approx(0.5) and r["mf"]["spearman"] == pytest.approx(0.7) and r["mf"]["partisan"] == pytest.approx(0.4) and r["recovery"] == pytest.approx(0.9)
               and r["converged"] == r["runs"] == C.SERIES_RUNS for r in rows)
    calls.clear()
    rows = ev.series_lam()
    assert len(calls) == len(C.SERIES_LAMIS) * C.SERIES_RUNS and {c[3] for c in calls} == {C.SERIES_SEED + 1000 + r for r in range(C.SERIES_RUNS)}
    assert [r["lam_i"] for r in rows] == list(C.SERIES_LAMIS) and {c[2] for c in calls} == set(C.SERIES_LAMIS) and {c[0] for c in calls} == {C.SERIES_SHARE_MAIN}
    calls.clear()
    rows = ev.series_gamma()
    assert len(calls) == len(C.SERIES_GAMMAS) * C.SERIES_RUNS and {c[3] for c in calls} == {C.SERIES_SEED + 2000 + r for r in range(C.SERIES_RUNS)}
    assert [r["gamma"] for r in rows] == list(C.SERIES_GAMMAS) and {c[1] for c in calls} == set(C.SERIES_GAMMAS)


def test_aggregation_zählt_nichtkonvergierte_läufe():
    ok = SimpleNamespace(metrics={k: {"spearman": 0.5, "partisan": 0.1, "quality": 1.0, "lean_a": 0.0} for k in ("mittel", "lager", "mf")}, recovery=0.5, fit={"converged": True})
    bad = SimpleNamespace(metrics=ok.metrics, recovery=1.0, fit={"converged": False})
    agg = ev._aggregate([ok, bad, ok])
    assert agg["converged"] == 2 and agg["runs"] == 3 and agg["recovery"] == pytest.approx(2 / 3)
