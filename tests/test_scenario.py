"""Szenario: erzeugte Welten (deterministisch aus dem Zufalls-Seed) und eingegebene Bewertungen."""

import numpy as np
import pytest

import brk_constants as C
import brk_scenario as S

HAND = "A 1 1 0 0\nA 1 1 0 0\nA 1 1 0 0\nA 1 0 1 0\nB 1 1 0 1\nB 0 1 0 1"


def test_welt_form_und_determinismus():
    w = S.make_world(120, 30, 70, 1.5, 60, 40, 3)
    assert w.ratings.shape == w.mask.shape == (120, 30) and w.camp.shape == (120,) and w.quality.shape == w.lean.shape == (30,)
    assert set(np.unique(w.ratings)) <= {0.0, 1.0} and set(np.unique(w.camp)) <= {-1, 1} and set(np.unique(w.lean)) <= {-1, 0, 1}
    w2 = S.make_world(120, 30, 70, 1.5, 60, 40, 3)
    assert np.array_equal(w.ratings, w2.ratings) and np.array_equal(w.mask, w2.mask) and np.array_equal(w.camp, w2.camp)
    assert not np.array_equal(w.ratings, S.make_world(120, 30, 70, 1.5, 60, 40, 4).ratings)


def test_anteile_lager_parteilichkeit_beobachtung():
    w = S.make_world(4000, 400, 80, 1.5, 60, 25, 1)
    assert np.mean(w.camp == 1) == pytest.approx(0.80, abs=0.02)
    assert np.mean(w.lean != 0) == pytest.approx(0.60, abs=0.06) and np.mean(w.lean == 1) == pytest.approx(0.30, abs=0.06)
    assert w.mask.mean() == pytest.approx(0.25, abs=0.01)
    assert w.quality.mean() == pytest.approx(0.0, abs=0.12) and w.quality.std() == pytest.approx(1.0, abs=0.08)


def test_zustimmungsrate_ohne_lagerneigung_und_mit():
    w = S.make_world(3000, 200, 50, 0.0, 60, 50, 2)
    # P(Qualität + Rauschen > 0,3) mit Standardabweichung √2: 1 − Φ(0,3 / √2) = 0,416
    assert w.ratings.mean() == pytest.approx(0.416, abs=0.02)
    g = S.make_world(3000, 200, 50, 1.5, 100, 50, 2)
    a_like = g.ratings[np.ix_(g.camp == 1, g.lean == 1)].mean()
    b_like = g.ratings[np.ix_(g.camp == -1, g.lean == 1)].mean()
    assert a_like > b_like + 0.3                                                      # Vorschläge mit Neigung zu A gefallen Lager A deutlich besser


@pytest.mark.parametrize("kw", [dict(users=0), dict(notes=0), dict(share=101), dict(share=-1), dict(gamma=-0.1), dict(partisan=101), dict(obs=0), dict(obs=101)])
def test_ungültige_weltparameter(kw):
    args = dict(users=50, notes=10, share=50, gamma=1.0, partisan=50, obs=50, seed=1)
    args.update(kw)
    with pytest.raises(ValueError):
        S.make_world(**args)


def test_eingabe_handbeispiel():
    w, err = S.parse_ratings(HAND)
    assert err is None and w.quality is None and w.lean is None
    assert w.camp.tolist() == [1, 1, 1, 1, -1, -1] and w.ratings.shape == (6, 4) and w.mask.all()
    assert w.ratings[3].tolist() == [1.0, 0.0, 1.0, 0.0] and w.ratings[5].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_eingabe_fehlende_bewertungen_komma_und_kleinbuchstaben():
    w, err = S.parse_ratings("a 1,0,-\nb 0 1 1\na - - 1\n")
    assert err is None and w.camp.tolist() == [1, -1, 1]
    assert w.mask.tolist() == [[True, True, False], [True, True, True], [False, False, True]] and w.ratings[0].tolist() == [1.0, 0.0, 0.0]


def test_eingabe_leer_ist_kein_fehler():
    assert S.parse_ratings("") == (None, None) and S.parse_ratings("  \n ") == (None, None) and S.parse_ratings(None) == (None, None)


@pytest.mark.parametrize("text", ["C 1 0\nA 1 0\nB 0 1", "A 1 2\nA 1 0\nB 0 1", "A 1 0\nA 1\nB 0 1", "A 1 0\nB 0 1", "A 1\nA 0\nB 1", "A - -\nB - -\nA - -",
                                  "\n".join(["A " + " ".join(["1"] * 31)] * 3)])
def test_eingabe_fehler(text):
    w, error = S.parse_ratings(text)
    assert w is None and error


def test_eingabe_grenzen():
    line = "A " + " ".join(["1"] * C.TEXT_MAX_NOTES)
    assert S.parse_ratings("\n".join([line] * C.TEXT_MAX_RATERS))[0] is not None
    assert S.parse_ratings("\n".join([line] * (C.TEXT_MAX_RATERS + 1)))[0] is None
    assert S.parse_ratings("\n".join([line] * C.TEXT_MIN_RATERS))[0] is not None
    two = "A 1 0"
    assert S.parse_ratings("\n".join([two] * C.TEXT_MIN_RATERS))[0] is not None                  # zwei Vorschläge genügen
    assert S.parse_ratings("\n".join(["A " + " ".join(["1"] * (C.TEXT_MAX_NOTES + 1))] * C.TEXT_MIN_RATERS))[0] is None
