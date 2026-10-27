"""Abbildungen: Zahl und Art der Spuren, gesperrte Achsen, Werte aus der Analyse und den Reihen."""

import pytest

import brk_constants as C
import brk_evaluation as ev
import brk_visualization as V

HAND = C.PRESETS["Handbeispiel (sechs Bewerter)"]["text"]


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_wertabbildung_erzeugter_welt():
    a = ev.analyse(ev.Settings(users=120, notes=40))
    fig = V.build_scores(a)
    assert _locked(fig) and len(fig.data) == 9                                             # drei Verfahren mal drei Neigungen
    assert [t.showlegend for t in fig.data] == [True] * 3 + [False] * 6
    first = fig.data[0]                                                                    # Verfahren Mittelwert, Neigung 0
    idx = [j for j in range(a.notes) if int(a.world.lean[j]) == 0]
    assert list(first.x) == pytest.approx([float(a.world.quality[j]) for j in idx]) and list(first.y) == pytest.approx([float(a.scores["mittel"][j]) for j in idx])
    top = {int(i) for i in a.tops["mittel"]}
    assert [c == "#111" for c in first.marker.line.color] == [j in top for j in idx]
    assert sum(len(t.x) for t in fig.data[:3]) == a.notes


def test_wertabbildung_eigener_bewertungen_sind_balken():
    a = ev.analyse(ev.Settings(text=HAND))
    fig = V.build_scores(a)
    assert _locked(fig) and len(fig.data) == 3 and all(t.type == "bar" and list(t.x) == ["V1", "V2", "V3", "V4"] for t in fig.data)
    assert list(fig.data[1].y) == pytest.approx([0.5, 0.75, 0.0, 0.0])


def test_faktorabbildung_getrennt_nach_lager():
    a = ev.analyse(ev.Settings(users=200, notes=40))
    fig = V.build_factor(a)
    assert _locked(fig) and [t.name for t in fig.data] == [f"Lager A ({int((a.world.camp == 1).sum())})", f"Lager B ({int((a.world.camp == -1).sum())})"]
    assert len(fig.data[0].y) + len(fig.data[1].y) == a.users


def test_zielfunktionsabbildung_fällt():
    a = ev.analyse(ev.Settings(users=100, notes=30))
    fig = V.build_loss(a)
    ys = list(fig.data[0].y)
    assert _locked(fig) and len(ys) == a.fit["steps"] + 1 and all(b <= x + 1e-9 for x, b in zip(ys, ys[1:])) and list(fig.data[0].x)[-1] == a.fit["steps"]


def _row(**kw):
    base = {"runs": 30, "recovery": 0.9, "converged": 30}
    for i, k in enumerate(("mittel", "lager", "mf")):
        base[k] = {"spearman": 0.5 + 0.1 * i, "partisan": 0.2 + 0.1 * i, "quality": 1.0, "lean_a": 0.0}
    base.update(kw)
    return base


def test_reihenabbildungen():
    share = V.build_share_series([_row(share=s) for s in C.SERIES_SHARES])
    assert _locked(share) and len(share.data) == 6 and share.data[0].yaxis == "y" and share.data[1].yaxis == "y2"
    assert list(share.data[0].x) == ["50 : 50", "60 : 40", "70 : 30", "80 : 20", "90 : 10"] and list(share.data[5].y) == pytest.approx([40.0] * 5)
    lam = V.build_lam_series([_row(lam_i=x) for x in C.SERIES_LAMIS])
    assert _locked(lam) and len(lam.data) == 3 and list(lam.data[0].x) == ["0,15", "0,5", "1", "2", "5", "10"] and list(lam.data[2].y) == pytest.approx([0.7] * 6)
    gamma = V.build_gamma_series([_row(gamma=x) for x in C.SERIES_GAMMAS])
    assert _locked(gamma) and list(gamma.data[1].x) == ["0,0", "0,5", "1,0", "1,5", "2,0", "3,0"]


def test_zahl_der_vorschläge_je_neigung():
    a = ev.analyse(ev.Settings(users=60, notes=50))
    counts = V.lean_counts(a)
    assert sum(counts.values()) == 50 and counts[0] == int((a.world.lean == 0).sum())
    assert V.lean_counts(ev.analyse(ev.Settings(text=HAND))) is None
