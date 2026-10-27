"""Kern: Rangverfahren für Vorschläge, die von Bewertern in zwei Lagern als hilfreich (1) oder nicht (0) bewertet werden; beobachtet ist nur ein Teil der Bewertungen (Maske). Drei Verfahren: mittlere Zustimmung, Mindestzustimmung über beide
Lager (braucht die Lagerangabe) und das brückenbasierte Verfahren: Faktorisierung r ≈ μ + i_Bewerter + i_Vorschlag + f_Bewerter · f_Vorschlag, wobei der Faktor die Lagerrichtung aufnimmt und der Intercept des Vorschlags (stark regularisiert) übrig bleibt;
nach ihm wird gerankt. Gelöst wird durch abwechselnde Ridge-Schritte (ALS): jeder Schritt ist die exakte Lösung eines Blocks, die Zielfunktion fällt also in jedem Schritt."""

import numpy as np

import brk_constants as C


def average_ranks(x):
    """Ränge mit Mittelwert bei Gleichstand (1 = kleinster Wert)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a, b) -> float:
    """Rangkorrelation; 0, wenn eine Seite konstant ist."""
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_scores(ratings, mask):
    """Mittlere Zustimmung je Vorschlag über die beobachteten Bewertungen; ohne Bewertung gilt das Gesamtmittel."""
    r, m = np.asarray(ratings, dtype=float), np.asarray(mask, dtype=bool)
    overall = float(r[m].mean()) if m.any() else 0.0
    count = m.sum(axis=0)
    total = np.where(m, r, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), overall)


def camp_scores(ratings, mask, camp):
    """Mittlere Zustimmung je Vorschlag in Lager A und in Lager B; hat ein Lager zu einem Vorschlag keine Bewertung (oder gar keine Bewerter), gilt für dieses Lager die mittlere Zustimmung aller Bewerter."""
    r, m, c = np.asarray(ratings, dtype=float), np.asarray(mask, dtype=bool), np.asarray(camp)
    overall = mean_scores(r, m)
    out = []
    for side in (1, -1):
        sel = c == side
        count = m[sel].sum(axis=0)
        total = np.where(m[sel], r[sel], 0.0).sum(axis=0)
        out.append(np.where(count > 0, total / np.maximum(count, 1), overall))
    return out[0], out[1]


def min_camp_scores(ratings, mask, camp):
    """Mindestzustimmung über die beiden Lager."""
    a, b = camp_scores(ratings, mask, camp)
    return np.minimum(a, b)


def _loss(r, mask, mu, iu, inn, fu, fn, lam_i, lam_f):
    """Zielfunktion elementweise (Referenz): Summe der Fehlerquadrate über die beobachteten Bewertungen plus Regularisierung."""
    pred = mu + iu[:, None] + inn[None, :] + fu[:, None] * fn[None, :]
    err = np.where(mask, r - pred, 0.0)
    return float((err**2).sum() + lam_i * ((iu**2).sum() + (inn**2).sum()) + lam_f * ((fu**2).sum() + (fn**2).sum()))


class _Sums:
    """Aus den Bewertungen einmal berechnete Größen, damit jeder ALS-Schritt nur Matrix-Vektor-Produkte braucht."""

    def __init__(self, r, m, mu):
        self.mf = m.astype(float)
        self.rm = self.mf * (r - mu)
        self.cnt_u, self.cnt_n = self.mf.sum(axis=1), self.mf.sum(axis=0)
        self.rm_u, self.rm_n = self.rm.sum(axis=1), self.rm.sum(axis=0)
        self.aa = float((self.rm * self.rm).sum())

    def loss(self, iu, inn, fu, fn, lam_i, lam_f):
        """Dieselbe Zielfunktion wie `_loss`, ohne Matrizen derselben Größe: Σ m·(a − b)² mit a = r − μ und b = i_u + i_n + f_u f_n, ausmultipliziert."""
        mf = self.mf
        cross = iu @ self.rm_u + inn @ self.rm_n + fu @ (self.rm @ fn)
        bb = ((self.cnt_u * iu**2).sum() + (self.cnt_n * inn**2).sum() + 2 * iu @ (mf @ inn) + 2 * ((iu * fu) @ (mf @ fn) + (fu @ mf) @ (inn * fn)) + (fu**2) @ (mf @ fn**2))
        return float(self.aa - 2 * cross + bb + lam_i * ((iu**2).sum() + (inn**2).sum()) + lam_f * ((fu**2).sum() + (fn**2).sum()))


def fit_bridging(ratings, mask, lam_i, lam_f, seed=0, max_iter=None, tol=None, init_sd=None):
    """Brückenbasierte Faktorisierung durch ALS. μ ist das Gesamtmittel der beobachteten Bewertungen (fest). Rückgabe: dict mit mu, iu, inn, fu, fn, losses (Zielfunktion nach jedem Schritt), steps, converged."""
    if lam_i <= 0 or lam_f <= 0:
        raise ValueError("Regularisierungen müssen positiv sein")
    r, m = np.asarray(ratings, dtype=float), np.asarray(mask, dtype=bool)
    if not m.any():
        raise ValueError("Keine beobachtete Bewertung")
    max_iter = C.MF_MAX_ITER if max_iter is None else max_iter
    tol = C.MF_TOL if tol is None else tol
    init_sd = C.MF_INIT_SD if init_sd is None else init_sd
    u, n = r.shape
    rng = np.random.default_rng(seed)
    mu = float(r[m].mean())
    iu, inn = np.zeros(u), np.zeros(n)
    fu, fn = rng.normal(0.0, init_sd, u), rng.normal(0.0, init_sd, n)
    sums = _Sums(r, m, mu)
    mf = sums.mf
    losses = [sums.loss(iu, inn, fu, fn, lam_i, lam_f)]
    converged = False
    for _ in range(max_iter):
        iu = (sums.rm_u - mf @ inn - fu * (mf @ fn)) / (sums.cnt_u + lam_i)
        inn = (sums.rm_n - iu @ mf - fn * (fu @ mf)) / (sums.cnt_n + lam_i)
        fu = (sums.rm @ fn - iu * (mf @ fn) - mf @ (inn * fn)) / (mf @ fn**2 + lam_f)
        fn = (fu @ sums.rm - (fu * iu) @ mf - (fu @ mf) * inn) / (fu**2 @ mf + lam_f)
        losses.append(sums.loss(iu, inn, fu, fn, lam_i, lam_f))
        if abs(losses[-2] - losses[-1]) <= tol * max(1.0, losses[-1]):
            converged = True
            break
    return {"mu": mu, "iu": iu, "inn": inn, "fu": fu, "fn": fn, "losses": losses, "steps": len(losses) - 1, "converged": converged}


def top_k(scores, k):
    """Indizes der k größten Werte, bei Gleichstand der kleinere Index zuerst."""
    scores = np.asarray(scores, dtype=float)
    return np.lexsort((np.arange(len(scores)), -scores))[: min(k, len(scores))]


def camp_recovery(user_factor, camp) -> float:
    """Betrag der Korrelation zwischen dem Bewerterfaktor und dem Lager (0, wenn eine Seite konstant ist); das Vorzeichen des Faktors ist beliebig."""
    f, c = np.asarray(user_factor, dtype=float), np.asarray(camp, dtype=float)
    if f.std() == 0 or c.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(f, c)[0, 1]))
