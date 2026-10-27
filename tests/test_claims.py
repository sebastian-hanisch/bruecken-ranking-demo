"""Befunde der festen Reihen: jede Aussage der README und der Verdikte steht hier als Prüfung. Die Reihen rechnen mit festen Zufalls-Seeds; geprüft werden Richtungen und grobe Bänder, nicht die letzte Stelle
(die Zufallsströme von numpy können sich zwischen Versionen unterscheiden)."""

import pytest

import brk_constants as C
import brk_evaluation as ev


@pytest.fixture(scope="module")
def shares():
    return ev.series_share()


@pytest.fixture(scope="module")
def lams():
    return ev.series_lam()


@pytest.fixture(scope="module")
def gammas():
    return ev.series_gamma()


def rho(r, k):
    return r[k]["spearman"]


def part(r, k):
    return r[k]["partisan"]


# --- Reihe 1: Lagergröße ----------------------------------------------------------------------------------------------------------------------


def test_reihe_1_aufbau(shares):
    assert [r["share"] for r in shares] == list(C.SERIES_SHARES) and all(r["runs"] == C.SERIES_RUNS and r["converged"] == C.SERIES_RUNS for r in shares)
    assert ev.share_table(shares)[0]["Lager"] == "50 : 50"


def test_bei_gleich_großen_lagern_genügt_der_mittelwert(shares):
    first = shares[0]
    assert rho(first, "mittel") > 0.9 and rho(first, "mf") > 0.9 and abs(rho(first, "mf") - rho(first, "mittel")) < 0.05                       # README: 0,94 und 0,95
    assert abs(part(first, "mf") - part(first, "mittel")) < 0.05


def test_mit_ungleichen_lagern_verzerrt_der_mittelwert_und_die_brücke_hilft(shares):
    mittel = [rho(r, "mittel") for r in shares]
    assert mittel == sorted(mittel, reverse=True) and mittel[-1] < 0.75                                                                        # README: 0,94 bis 0,69
    assert all(rho(r, "mf") > 0.9 for r in shares) and rho(shares[-1], "mf") - mittel[-1] > 0.15
    parts = [part(r, "mittel") for r in shares]
    assert parts == sorted(parts) and parts[-1] > 0.65 and parts[0] < 0.4                                                                      # README: 34 % bis 76 %
    assert all(part(r, "mf") < 0.45 for r in shares) and part(shares[-1], "mittel") - part(shares[-1], "mf") > 0.25


def test_mindestlager_senkt_parteiisch_am_stärksten_verliert_aber_rangtreue(shares):
    for r in shares:
        assert part(r, "lager") < 0.25 and part(r, "lager") < part(r, "mf")
    assert rho(shares[0], "lager") < rho(shares[0], "mf") - 0.1                                                                                # README: 0,78 gegen 0,95 bei 50 : 50
    assert rho(shares[-1], "lager") < rho(shares[-1], "mf") - 0.1


def test_der_bewerterfaktor_findet_die_lager(shares):
    assert all(r["recovery"] > 0.9 for r in shares)


def test_verdikt_reihe_1(shares):
    kind, text = ev.verdict_share(shares)
    assert kind == "info" and "(50 : 50)" in text and "bei 90 : 10 fällt der Mittelwert" in text


# --- Reihe 2: Regularisierung -----------------------------------------------------------------------------------------------------------------


def test_reihe_2_aufbau(lams):
    assert [r["lam_i"] for r in lams] == list(C.SERIES_LAMIS) and ev.lam_table(lams)[0]["λ Intercept"] == "0,15"
    assert all(r["converged"] == C.SERIES_RUNS for r in lams)


def test_mittelwert_und_mindestlager_sind_von_lambda_unberührt(lams):
    assert len({round(rho(r, "mittel"), 9) for r in lams}) == 1 and len({round(rho(r, "lager"), 9) for r in lams}) == 1                      # dieselben Welten in jeder Zeile


def test_schwache_regularisierung_lässt_die_brücke_beim_mittelwert(lams):
    weak = lams[0]
    assert rho(weak, "mf") < rho(weak, "mittel") + 0.1 and part(weak, "mf") > 0.4                                                              # README: 0,80 gegen 0,77, 50 % parteiisch


def test_starke_regularisierung_bringt_die_brücke_zum_ziel(lams):
    rhos = [rho(r, "mf") for r in lams]
    assert all(b >= a - 0.02 for a, b in zip(rhos, rhos[1:])) and rhos[-1] > 0.9 and rhos[-1] - rhos[0] > 0.1
    assert part(lams[-1], "mf") < 0.35 and part(lams[-1], "mf") < part(lams[0], "mf") - 0.1


def test_verdikt_reihe_2(lams):
    kind, text = ev.verdict_lam(lams)
    assert kind == "success" and "(λ = 0,15)" in text and "mit λ = 10,0" in text


# --- Reihe 3: Lagerneigung --------------------------------------------------------------------------------------------------------------------


def test_reihe_3_aufbau(gammas):
    assert [r["gamma"] for r in gammas] == list(C.SERIES_GAMMAS) and ev.gamma_table(gammas)[0]["Lagerneigung"] == "0,0"


def test_ohne_lagerneigung_sind_alle_verfahren_gleich_gut_und_der_faktor_findet_nichts(gammas):
    first = gammas[0]
    assert all(rho(first, k) > 0.95 for k in ("mittel", "lager", "mf")) and first["recovery"] < 0.15                                           # README: 0,99 / 0,97 / 0,98, Lager im Faktor 4 %


def test_mit_wachsender_neigung_sinkt_der_mittelwert_und_die_brücke_bleibt_vorn(gammas):
    mittel = [rho(r, "mittel") for r in gammas]
    assert mittel == sorted(mittel, reverse=True) and mittel[-1] < 0.6
    for r in gammas[2:]:
        assert rho(r, "mf") > rho(r, "mittel") + 0.05
    assert rho(gammas[-1], "mf") - rho(gammas[-1], "mittel") > 0.1 and all(r["recovery"] > 0.9 for r in gammas[3:])


def test_verdikt_reihe_3(gammas):
    kind, text = ev.verdict_gamma(gammas)
    assert kind == "info" and "Ohne Lagerneigung" in text and "bei Neigung 3,0" in text
