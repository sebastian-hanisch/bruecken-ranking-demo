"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import brk_constants as C

RANKER_COLORS = {"mittel": C.COLOR_MEAN, "lager": C.COLOR_CAMP, "mf": C.COLOR_MF}
LEAN_COLORS = {1: C.COLOR_LEAN_A, -1: C.COLOR_LEAN_B, 0: C.COLOR_NEUTRAL}
LEAN_NAMES = {1: "neigt zu Lager A", -1: "neigt zu Lager B", 0: "neutral"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scores(a):
    """Bei erzeugten Welten: je Verfahren die Bewertung jedes Vorschlags über seiner wahren Qualität, Farbe nach Lagerneigung, Rand für die besten Vorschläge.
    Bei eingegebenen Bewertungen: Balken der Werte je Vorschlag."""
    titles = [C.RANKER_SHORT[k] for k in ("mittel", "lager", "mf")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.08)
    keys = ("mittel", "lager", "mf")
    if a.world.quality is None:
        xs = [f"V{j + 1}" for j in range(a.notes)]
        for col, k in enumerate(keys, start=1):
            fig.add_trace(go.Bar(x=xs, y=[float(v) for v in a.scores[k]], marker_color=RANKER_COLORS[k], showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra>" + C.RANKER_SHORT[k] + "</extra>"), row=1, col=col)
        fig.update_xaxes(type="category")
        return _base(fig, height=320)
    top_sets = {k: set(int(i) for i in a.tops[k]) for k in keys}
    for col, k in enumerate(keys, start=1):
        for lean in (0, 1, -1):
            idx = [j for j in range(a.notes) if int(a.world.lean[j]) == lean]
            fig.add_trace(go.Scatter(
                x=[float(a.world.quality[j]) for j in idx], y=[float(a.scores[k][j]) for j in idx], mode="markers", name=LEAN_NAMES[lean], legendgroup=str(lean), showlegend=(col == 1),
                marker=dict(color=LEAN_COLORS[lean], size=8, line=dict(color=["#111" if j in top_sets[k] else "rgba(0,0,0,0)" for j in idx], width=2)),
                text=[f"Vorschlag {j + 1}: Qualität {a.world.quality[j]:.2f}, {LEAN_NAMES[lean]}" for j in idx], hovertemplate="%{text}<extra>" + C.RANKER_SHORT[k] + "</extra>"), row=1, col=col)
        fig.update_xaxes(title_text="wahre Qualität", row=1, col=col)
    fig.update_yaxes(title_text="Wert des Verfahrens", row=1, col=1)
    return _base(fig, height=380)


def build_factor(a):
    """Bewerterfaktor des brückenbasierten Verfahrens, getrennt nach Lager: die Lagerrichtung steckt im Faktor, nicht im Intercept."""
    fig = go.Figure()
    for camp, name, color in ((1, "Lager A", C.COLOR_LEAN_A), (-1, "Lager B", C.COLOR_LEAN_B)):
        f = a.fit["fu"][a.world.camp == camp]
        fig.add_trace(go.Box(y=[float(v) for v in f], name=f"{name} ({len(f)})", marker_color=color, boxpoints="all", jitter=0.4, pointpos=0, hovertemplate="%{y:.3f}<extra>" + name + "</extra>"))
    fig.update_yaxes(title_text="Bewerterfaktor")
    return _base(fig, height=300)


def build_loss(a):
    """Zielfunktion der Faktorisierung über die Schritte (fällt in jedem Schritt)."""
    losses = a.fit["losses"]
    fig = go.Figure(go.Scatter(x=list(range(len(losses))), y=[float(v) for v in losses], mode="lines", line=dict(color=C.COLOR_MF, width=2.5), hovertemplate="Schritt %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="Zielfunktion")
    return _base(fig, height=260)


def _lines(rows, xkey, xlabel, ykey, ylabel, height=360, fmt=None):
    fig = go.Figure()
    xs = [fmt(r[xkey]) if fmt else str(r[xkey]) for r in rows]
    for k in ("mittel", "lager", "mf"):
        fig.add_trace(go.Scatter(x=xs, y=[r[k][ykey] for r in rows], mode="lines+markers", name=C.RANKER_SHORT[k], line=dict(color=RANKER_COLORS[k], width=2.5),
                                 hovertemplate="%{x}: %{y:.2f}<extra>" + C.RANKER_SHORT[k] + "</extra>"))
    fig.update_xaxes(title_text=xlabel, type="category")
    fig.update_yaxes(title_text=ylabel)
    return _base(fig, height=height)


def build_share_series(rows):
    """Oben die Rangkorrelation zur Qualität, unten der Anteil parteiischer Vorschläge unter den besten zehn, über der Größe des großen Lagers."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.5, 0.5])
    xs = [f"{r['share']} : {100 - r['share']}" for r in rows]
    for k in ("mittel", "lager", "mf"):
        fig.add_trace(go.Scatter(x=xs, y=[r[k]["spearman"] for r in rows], mode="lines+markers", name=C.RANKER_SHORT[k], legendgroup=k, line=dict(color=RANKER_COLORS[k], width=2.5),
                                 hovertemplate="%{x}: %{y:.2f}<extra>" + C.RANKER_SHORT[k] + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[k]["partisan"] for r in rows], mode="lines+markers", name=C.RANKER_SHORT[k], legendgroup=k, showlegend=False, line=dict(color=RANKER_COLORS[k], width=2.5, dash="dot"),
                                 hovertemplate="%{x}: %{y:.0f} %<extra>" + C.RANKER_SHORT[k] + "</extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Rangkorrelation zur Qualität", row=1, col=1)
    fig.update_yaxes(title_text="Parteiisch unter den besten (%)", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Lagergrößen", type="category", row=2, col=1)
    return _base(fig, height=470)


def build_lam_series(rows):
    """Rangkorrelation zur Qualität über der Regularisierung des Intercepts (Mittelwert und Mindestlager sind davon unberührt)."""
    return _lines(rows, "lam_i", "Regularisierung des Intercepts λ", "spearman", "Rangkorrelation zur Qualität", fmt=lambda x: f"{x:g}".replace(".", ","))


def build_gamma_series(rows):
    """Rangkorrelation zur Qualität über der Stärke der Lagerneigung."""
    return _lines(rows, "gamma", "Lagerneigung", "spearman", "Rangkorrelation zur Qualität", fmt=lambda x: f"{x:.1f}".replace(".", ","))


def lean_counts(a):
    """Zahl der Vorschläge je Lagerneigung (für Beschriftungen und Tests)."""
    if a.world.lean is None:
        return None
    return {lean: int(np.sum(a.world.lean == lean)) for lean in (1, 0, -1)}
