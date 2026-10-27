"""Orakel: unabhängige Gegenproben. Die Zielfunktion (schnell, ohne große Matrizen) gegen die elementweise Referenz und gegen eine Rechnung in Schleifen; jeder ALS-Schritt gegen eine zweite Umsetzung in reinen Python-Schleifen
und gegen die Normalgleichungen der Ridge-Regression (numpy.linalg.solve); stationärer Punkt per Gradient; Rangkorrelation gegen scipy; Mittel und Mindestzustimmung gegen Schleifen; ein rauschfreier Rang-1-Fall."""

import numpy as np
import pytest
from scipy.stats import spearmanr

import brk_algorithm as A
from brk_scenario import make_world


def random_problem(seed, u=14, n=9, density=0.6):
    rng = np.random.default_rng(seed)
    return (rng.random((u, n)) < 0.5).astype(float), rng.random((u, n)) < density


# --- Zielfunktion -----------------------------------------------------------------------------------------------------------------------------


def loop_loss(r, m, mu, iu, inn, fu, fn, lam_i, lam_f):
    total = 0.0
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if m[i, j]:
                total += (r[i, j] - mu - iu[i] - inn[j] - fu[i] * fn[j]) ** 2
    return total + lam_i * (sum(x * x for x in iu) + sum(x * x for x in inn)) + lam_f * (sum(x * x for x in fu) + sum(x * x for x in fn))


def test_schnelle_zielfunktion_gegen_referenz_und_schleifen_bei_beliebigen_werten():
    rng = np.random.default_rng(1)
    for seed in range(10):
        r, m = random_problem(seed)
        mu = float(r[m].mean())
        iu, inn, fu, fn = rng.normal(size=14), rng.normal(size=9), rng.normal(size=14), rng.normal(size=9)
        fast = A._Sums(r, m, mu).loss(iu, inn, fu, fn, 0.7, 0.2)
        assert fast == pytest.approx(A._loss(r, m, mu, iu, inn, fu, fn, 0.7, 0.2), rel=1e-10)
        assert fast == pytest.approx(loop_loss(r, m, mu, iu, inn, fu, fn, 0.7, 0.2), rel=1e-10)


# --- ALS-Schritte gegen zweite Umsetzung ------------------------------------------------------------------------------------------------------


def loop_als(r, m, lam_i, lam_f, iu, inn, fu, fn, steps):
    """Dieselbe Vorschrift in Schleifen (jeder Block eine exakte Ridge-Lösung bei festen anderen Blöcken)."""
    u, n = r.shape
    mu = float(r[m].mean())
    iu, inn, fu, fn = list(iu), list(inn), list(fu), list(fn)
    for _ in range(steps):
        for i in range(u):
            num = sum(r[i, j] - mu - inn[j] - fu[i] * fn[j] for j in range(n) if m[i, j])
            iu[i] = num / (sum(1 for j in range(n) if m[i, j]) + lam_i)
        for j in range(n):
            num = sum(r[i, j] - mu - iu[i] - fu[i] * fn[j] for i in range(u) if m[i, j])
            inn[j] = num / (sum(1 for i in range(u) if m[i, j]) + lam_i)
        for i in range(u):
            num = sum((r[i, j] - mu - iu[i] - inn[j]) * fn[j] for j in range(n) if m[i, j])
            fu[i] = num / (sum(fn[j] ** 2 for j in range(n) if m[i, j]) + lam_f)
        for j in range(n):
            num = sum((r[i, j] - mu - iu[i] - inn[j]) * fu[i] for i in range(u) if m[i, j])
            fn[j] = num / (sum(fu[i] ** 2 for i in range(u) if m[i, j]) + lam_f)
    return np.array(iu), np.array(inn), np.array(fu), np.array(fn)


def test_als_gegen_schleifenumsetzung():
    for seed in range(6):
        r, m = random_problem(seed)
        steps = 7
        f = A.fit_bridging(r, m, 0.8, 0.1, seed=seed, max_iter=steps, tol=0.0)
        init = np.random.default_rng(seed)
        fu0, fn0 = init.normal(0.0, 1.0, 14), init.normal(0.0, 1.0, 9)
        iu, inn, fu, fn = loop_als(r, m, 0.8, 0.1, np.zeros(14), np.zeros(9), fu0, fn0, steps)
        assert f["iu"] == pytest.approx(iu, abs=1e-10) and f["inn"] == pytest.approx(inn, abs=1e-10)
        assert f["fu"] == pytest.approx(fu, abs=1e-10) and f["fn"] == pytest.approx(fn, abs=1e-10)


def test_vorschlagsfaktor_gegen_normalgleichung():
    """Der letzte Block eines Schritts für sich: der Faktor eines Vorschlags ist die Lösung der Ridge-Normalgleichung (X'X + λI) g = X'y mit X = den Faktoren der beobachtenden Bewerter, bei den Endwerten der anderen Blöcke."""
    r, m = random_problem(3)
    f = A.fit_bridging(r, m, 0.8, 0.1, seed=3, max_iter=4, tol=0.0)
    for j in range(r.shape[1]):
        rows = np.flatnonzero(m[:, j])
        x = f["fu"][rows][:, None]
        y = r[rows, j] - f["mu"] - f["iu"][rows] - f["inn"][j]
        solution = np.linalg.solve(x.T @ x + 0.1 * np.eye(1), x.T @ y)
        assert f["fn"][j] == pytest.approx(float(solution[0]), abs=1e-10)


def test_stationärer_punkt_per_gradient():
    r, m = random_problem(2, u=20, n=10, density=0.7)
    f = A.fit_bridging(r, m, 1.0, 0.1, seed=2, max_iter=3000, tol=1e-14)
    assert f["converged"]
    names = ("iu", "inn", "fu", "fn")
    base = {k: f[k].copy() for k in names}
    loss0 = A._loss(r, m, f["mu"], base["iu"], base["inn"], base["fu"], base["fn"], 1.0, 0.1)
    for key in names:
        for idx in range(len(base[key])):
            plus = {k: v.copy() for k, v in base.items()}
            plus[key][idx] += 1e-5
            minus = {k: v.copy() for k, v in base.items()}
            minus[key][idx] -= 1e-5
            lp = A._loss(r, m, f["mu"], plus["iu"], plus["inn"], plus["fu"], plus["fn"], 1.0, 0.1)
            lm = A._loss(r, m, f["mu"], minus["iu"], minus["inn"], minus["fu"], minus["fn"], 1.0, 0.1)
            assert abs((lp - lm) / 2e-5) < 1e-4
    assert loss0 == pytest.approx(f["losses"][-1], rel=1e-9)


# --- Rangkorrelation, Mittel, Mindestzustimmung -----------------------------------------------------------------------------------------------


def test_rangkorrelation_gegen_scipy_mit_gleichständen():
    rng = np.random.default_rng(4)
    for _ in range(40):
        a = rng.integers(0, 6, size=25).astype(float)
        b = rng.integers(0, 6, size=25).astype(float)
        if a.std() == 0 or b.std() == 0:
            continue
        assert A.spearman(a, b) == pytest.approx(float(spearmanr(a, b)[0]), abs=1e-12)


def test_mittel_und_lagerzustimmung_gegen_schleifen():
    for seed in range(6):
        w = make_world(40, 12, 70, 1.5, 60, 40, seed)
        r, m, camp = w.ratings, w.mask, w.camp
        mean = A.mean_scores(r, m)
        a, b = A.camp_scores(r, m, camp)
        for j in range(12):
            obs = [r[i, j] for i in range(40) if m[i, j]]
            overall = float(np.mean(r[m]))
            assert mean[j] == pytest.approx(np.mean(obs) if obs else overall)
            for side, got in ((1, a), (-1, b)):
                side_obs = [r[i, j] for i in range(40) if m[i, j] and camp[i] == side]
                assert got[j] == pytest.approx(np.mean(side_obs) if side_obs else mean[j])
        assert A.min_camp_scores(r, m, camp) == pytest.approx(np.minimum(a, b))


# --- Rauschfreier Rang-1-Fall -----------------------------------------------------------------------------------------------------------------


def test_exakt_zerlegbare_bewertungen_werden_reproduziert():
    rng = np.random.default_rng(6)
    iu0, in0, fu0, fn0 = rng.normal(size=8), rng.normal(size=6), rng.normal(size=8), rng.normal(size=6)
    r = 0.3 + iu0[:, None] + in0[None, :] + fu0[:, None] * fn0[None, :]
    m = np.ones_like(r, dtype=bool)
    f = A.fit_bridging(r, m, 1e-9, 1e-9, seed=1, max_iter=20000, tol=1e-16)
    pred = f["mu"] + f["iu"][:, None] + f["inn"][None, :] + f["fu"][:, None] * f["fn"][None, :]
    assert np.abs(pred - r).max() < 1e-3
