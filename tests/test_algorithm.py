"""Kern: Ränge, Rangkorrelation, Mittel und Mindestzustimmung, Faktorisierung (ALS), Rangliste, Lagererkennung. Jede Einheit an einer Mini-Instanz mit Handrechnung (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import brk_algorithm as A
import brk_constants as C

# --- Ränge und Rangkorrelation ----------------------------------------------------------------------------------------------------------------


def test_ränge_mit_mittelwert_bei_gleichstand():
    assert A.average_ranks([3.0, 1.0, 2.0]).tolist() == [3.0, 1.0, 2.0]
    assert A.average_ranks([1.0, 1.0, 2.0]).tolist() == [1.5, 1.5, 3.0]
    assert A.average_ranks([5.0, 5.0, 5.0, 1.0]).tolist() == [3.0, 3.0, 3.0, 1.0]


def test_rangkorrelation_handrechnung():
    assert A.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)             # 1 − 6 · 2 / (4 · 15)
    assert A.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_rangkorrelation_mit_konstanter_seite_ist_null():
    assert A.spearman([1, 1, 1], [1, 2, 3]) == 0.0 and A.spearman([1, 2, 3], [4, 4, 4]) == 0.0


# --- Mittel und Mindestzustimmung -------------------------------------------------------------------------------------------------------------


def test_mittlere_zustimmung_handbeispiel(hand):
    camp, r, m = hand
    assert A.mean_scores(r, m) == pytest.approx([5 / 6, 5 / 6, 1 / 6, 1 / 3])


def test_mittel_nur_über_beobachtete_bewertungen_und_ohne_bewertung_das_gesamtmittel():
    r = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    m = np.array([[True, False, False], [True, True, False], [False, False, False]])
    # Vorschlag 1: Bewerter 1 und 2 (1 und 0) → 0,5; Vorschlag 2: nur Bewerter 2 (1) → 1; Vorschlag 3: keine Bewertung → Gesamtmittel (1 + 0 + 1) / 3
    assert A.mean_scores(r, m) == pytest.approx([0.5, 1.0, 2 / 3])


def test_zustimmung_je_lager_und_mindestzustimmung(hand):
    camp, r, m = hand
    a, b = A.camp_scores(r, m, camp)
    assert a == pytest.approx([1.0, 0.75, 0.25, 0.0]) and b == pytest.approx([0.5, 1.0, 0.0, 1.0])
    assert A.min_camp_scores(r, m, camp) == pytest.approx([0.5, 0.75, 0.0, 0.0])


def test_lager_ohne_bewertung_nimmt_das_gesamtmittel():
    r = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    m = np.array([[True, True], [True, True], [True, False]])
    camp = np.array([1, 1, -1])
    a, b = A.camp_scores(r, m, camp)
    assert a == pytest.approx([1.0, 0.5]) and b == pytest.approx([0.0, 0.5])                 # Lager B hat zu Vorschlag 2 keine Bewertung: Gesamtmittel 0,5
    only_a, only_b = A.camp_scores(r, m, np.array([1, 1, 1]))
    assert only_b == pytest.approx(A.mean_scores(r, m))                                       # gar kein Bewerter in Lager B


# --- Rangliste und Lagererkennung -------------------------------------------------------------------------------------------------------------


def test_beste_vorschläge_mit_gleichstand_nach_index():
    assert A.top_k([0.5, 0.9, 0.9, 0.1], 3).tolist() == [1, 2, 0]
    assert A.top_k([0.5, 0.5, 0.5], 2).tolist() == [0, 1]
    assert A.top_k([1.0, 2.0], 10).tolist() == [1, 0]


def test_lagererkennung_betrag_der_korrelation():
    camp = np.array([1, 1, -1, -1])
    assert A.camp_recovery([0.3, 0.5, -0.4, -0.2], camp) > 0.95
    assert A.camp_recovery([-0.3, -0.5, 0.4, 0.2], camp) == pytest.approx(A.camp_recovery([0.3, 0.5, -0.4, -0.2], camp))     # Vorzeichen beliebig
    assert A.camp_recovery([0.1, 0.1, 0.1, 0.1], camp) == 0.0 and A.camp_recovery([0.1, 0.2, 0.3, 0.4], np.ones(4)) == 0.0


# --- Faktorisierung ---------------------------------------------------------------------------------------------------------------------------


def _small():
    rng = np.random.default_rng(5)
    return (rng.random((12, 7)) < 0.5).astype(float), rng.random((12, 7)) < 0.7


def test_faktorisierung_form_und_determinismus():
    r, m = _small()
    f = A.fit_bridging(r, m, 1.0, 0.1, seed=3)
    assert f["iu"].shape == (12,) and f["inn"].shape == (7,) and f["fu"].shape == (12,) and f["fn"].shape == (7,)
    assert f["mu"] == pytest.approx(r[m].mean()) and f["steps"] == len(f["losses"]) - 1
    g = A.fit_bridging(r, m, 1.0, 0.1, seed=3)
    assert np.array_equal(f["inn"], g["inn"]) and f["steps"] == g["steps"]
    assert not np.array_equal(f["fu"], A.fit_bridging(r, m, 1.0, 0.1, seed=4)["fu"])           # anderer Startwert


def test_zielfunktion_fällt_in_jedem_schritt():
    r, m = _small()
    losses = A.fit_bridging(r, m, 0.5, 0.05, seed=1, max_iter=60, tol=0.0)["losses"]
    assert len(losses) == 61 and all(b <= a + 1e-9 for a, b in zip(losses, losses[1:])) and losses[-1] < losses[0]


def test_abbruch_bei_toleranz_und_bei_höchstzahl():
    r, m = _small()
    f = A.fit_bridging(r, m, 1.0, 0.1, seed=1, tol=1e-3)
    assert f["converged"] and f["steps"] < C.MF_MAX_ITER
    g = A.fit_bridging(r, m, 1.0, 0.1, seed=1, max_iter=5, tol=0.0)
    assert not g["converged"] and g["steps"] == 5
    h = A.fit_bridging(r, m, 1.0, 0.1, seed=1, max_iter=0)
    assert h["steps"] == 0 and not h["converged"] and len(h["losses"]) == 1


def test_standardwerte_der_faktorisierung(monkeypatch):
    r, m = _small()
    monkeypatch.setattr(C, "MF_MAX_ITER", 3)
    assert A.fit_bridging(r, m, 1.0, 0.1, seed=1, tol=0.0)["steps"] == 3
    monkeypatch.setattr(C, "MF_INIT_SD", 0.0)
    f = A.fit_bridging(r, m, 1.0, 0.1, seed=1, tol=0.0)
    assert np.abs(f["fu"]).max() == 0.0 and np.abs(f["fn"]).max() == 0.0                       # Startwerte null bleiben null (Fixpunkt ohne Faktor)


def test_ungültige_eingaben_der_faktorisierung():
    r, m = _small()
    with pytest.raises(ValueError):
        A.fit_bridging(r, m, 0.0, 0.1)
    with pytest.raises(ValueError):
        A.fit_bridging(r, m, 1.0, -0.1)
    with pytest.raises(ValueError):
        A.fit_bridging(r, m, 1.0, 0.0)                                                        # genau null ist ebenfalls ungültig
    with pytest.raises(ValueError):
        A.fit_bridging(r, np.zeros_like(m), 1.0, 0.1)


def test_intercept_ohne_faktor_ist_die_geschrumpfte_mittlere_abweichung():
    # Ohne Faktor (Startwerte 0) ist iu = Σ_n (r − μ − i_n) / (Anzahl + λ), i_n entsprechend; Vorschlag 1 mit nur einer Bewertung wird zur Mitte gezogen
    r = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    m = np.array([[True, True], [True, True], [False, True]])
    f = A.fit_bridging(r, m, 2.0, 0.1, seed=0, init_sd=0.0, max_iter=1, tol=0.0)
    mu = r[m].mean()                                                                           # 4 von 5
    iu = np.array([(1 - mu + 1 - mu), (0 - mu + 1 - mu), (1 - mu)]) / (np.array([2.0, 2.0, 1.0]) + 2.0)
    inn = np.array([(1 - mu - iu[0]) + (0 - mu - iu[1]), (1 - mu - iu[0]) + (1 - mu - iu[1]) + (1 - mu - iu[2])]) / (np.array([2.0, 3.0]) + 2.0)
    assert f["iu"] == pytest.approx(iu) and f["inn"] == pytest.approx(inn)
