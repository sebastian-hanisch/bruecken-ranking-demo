"""Brückenbasiertes Ranking: Zustimmung über Lager hinweg - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Gelber Kandidat der Linie „Abstimmen und Fair Teilen“: Bewerter in zwei Lagern bewerten Vorschläge als hilfreich oder nicht. Die mittlere Zustimmung belohnt, was das größere Lager mag; die Mindestzustimmung über beide Lager braucht die Lagerangabe;
das brückenbasierte Verfahren faktorisiert die Bewertungen, lässt die Lagerrichtung im Faktor und rankt nach dem übrigen Intercept des Vorschlags. Der Kern liegt in brk_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import brk_constants as C
import brk_evaluation as ev
from brk_evaluation import Settings, analyse, num
from brk_layout import dunkles_design, kopf_css
from brk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from brk_scenario import parse_ratings
from brk_visualization import build_factor, build_gamma_series, build_lam_series, build_loss, build_scores, build_share_series

st.set_page_config(page_title="Brückenbasiertes Ranking – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series_share():
    return ev.series_share()


@st.cache_data(show_spinner=False)
def _series_lam():
    return ev.series_lam()


@st.cache_data(show_spinner=False)
def _series_gamma():
    return ev.series_gamma()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in brk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌉 Brückenbasiertes Ranking: Zustimmung über Lager hinweg")
st.markdown(
    """
Eine Plattform lässt **Vorschläge** von Nutzern bewerten (**hilfreich** oder **nicht hilfreich**) und zeigt die besten zuerst. Gehören die Bewerter **zwei Lagern** an, belohnt die **mittlere Zustimmung**, was das **größere Lager** mag, und mit einer Neigung zu einem Lager sind das oft parteiische Vorschläge statt der guten.
Das **brückenbasierte Verfahren** zerlegt die Bewertungen in einen **Lagerfaktor** (wer wie nah an wem ist) und einen **Intercept je Vorschlag**, der übrig bleibt, wenn man die Lagerrichtung herausrechnet: nach ihm wird gerankt. Die Demo vergleicht drei Verfahren gegen die **wahre Qualität** der Vorschläge, die in der Simulation bekannt ist, und zeigt, **wann** die Brücke hilft und **wovon** das abhängt.
Verwandt: `nmf-demo` (Faktorisierung), `meinungsbildung-demo` (Meinungsbildung im Netz) und `quadratisches-waehlen-demo` (Mehrheit und Intensität).

Weiter unten: die **Kernfrage**, der Abschnitt **🔬 Feste Reihen** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wird gerechnet", expanded=True):
    st.markdown(
        """
1. **Welt.** Bewerter gehören Lager A oder B an; Vorschläge haben eine wahre **Qualität** und eine **Lagerneigung** (A, B oder neutral). Ein Bewerter findet einen Vorschlag hilfreich, wenn Qualität + Lagerneigung · Lager + Rauschen eine Schwelle übersteigt. Nur ein Teil der Bewertungen wird beobachtet.
2. **Mittlere Zustimmung.** Je Vorschlag der Anteil „hilfreich“ unter den beobachteten Bewertungen.
3. **Mindestzustimmung über beide Lager.** Je Vorschlag die kleinere der mittleren Zustimmungen der beiden Lager; das braucht die Lagerangabe jedes Bewerters.
4. **Brückenbasiert.** Bewertung ≈ μ + Intercept des Bewerters + **Intercept des Vorschlags** + Faktor des Bewerters · Faktor des Vorschlags. Die Intercepts sind stark, die Faktoren schwach regularisiert; gelöst wird abwechselnd (ALS), jeder Schritt ist eine exakte Ridge-Regression. Gerankt wird nach dem Intercept des Vorschlags, die Lager muss das Verfahren selbst finden.
5. **Maße.** Rangkorrelation (Spearman) der Werte mit der wahren Qualität; Anteil parteiischer Vorschläge unter den besten zehn; mittlere Qualität der besten zehn; wie stark der Bewerterfaktor dem Lager folgt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

_world_valid, _ = parse_ratings(st.session_state["text_input_area"])
custom = _world_valid is not None
with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    users = st.slider("Zahl der Bewerter", *bounds("users_slider"), key="users_slider", disabled=custom, help="Wie viele Nutzer bewerten. Bei eigenen Bewertungen bestimmt die Zahl der Zeilen den Wert.")
    notes = st.slider("Zahl der Vorschläge", *bounds("notes_slider"), key="notes_slider", disabled=custom, help="Wie viele Vorschläge es gibt. Bei eigenen Bewertungen bestimmt die Zahl der Spalten den Wert.")
    share = st.slider("Größe des großen Lagers (%)", *bounds("share_slider"), key="share_slider", disabled=custom, help="Anteil der Bewerter in Lager A. 50 = gleich große Lager.")
    gamma = st.slider("Lagerneigung", *bounds("gamma_slider"), key="gamma_slider", step=0.1, disabled=custom, help="Wie stark das Lager die Zustimmung zu parteiischen Vorschlägen bestimmt. 0: das Lager spielt keine Rolle.")
    partisan = st.slider("Parteiische Vorschläge (%)", *bounds("partisan_slider"), key="partisan_slider", disabled=custom, help="Anteil der Vorschläge, die zu einem der Lager neigen (je zur Hälfte A und B).")
    obs = st.slider("Beobachtete Bewertungen (%)", *bounds("obs_slider"), key="obs_slider", disabled=custom, help="Welcher Anteil aller möglichen Bewertungen vorliegt.")
    st.markdown("**Brückenverfahren**")
    lam_i = st.slider("Regularisierung der Intercepts λ", *bounds("lami_slider"), key="lami_slider", step=0.05, help="Wie stark die Intercepts (Bewerter und Vorschläge) zur Mitte gezogen werden. Je größer, desto weniger Lagerrichtung bleibt im Intercept des Vorschlags.")
    lam_f = st.slider("Regularisierung der Faktoren", *bounds("lamf_slider"), key="lamf_slider", step=0.01, help="Wie stark die Faktoren zu null gezogen werden. Schwach, damit der Faktor die Lagerrichtung aufnehmen kann.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Welt und Startwerte der Faktorisierung fest.")
    st.button("🎲 Neues Szenario erzeugen", width="stretch", on_click=randomize_seed)
    st.markdown("**Eigene Bewertungen (optional)**")
    text = st.text_area("Eine Zeile je Bewerter", key="text_input_area", height=120, placeholder="z. B.\nA 1 1 0 0\nB 1 0 1 -\nA 0 1 1 1",
                        help=f"Je Zeile: das Lager (A oder B), dann je Vorschlag 1 (hilfreich), 0 (nicht hilfreich) oder - (nicht bewertet), getrennt durch Leerzeichen. {C.TEXT_MIN_RATERS} bis {C.TEXT_MAX_RATERS} Bewerter, {C.TEXT_MIN_NOTES} bis {C.TEXT_MAX_NOTES} Vorschläge. Leer lassen, um die Welt erzeugen zu lassen. Qualität und Neigung der Vorschläge sind dann unbekannt.")

sync_query_params({"users_slider": int(users), "notes_slider": int(notes), "share_slider": int(share), "gamma_slider": round(float(gamma), 1), "partisan_slider": int(partisan), "obs_slider": int(obs),
                   "lami_slider": round(float(lam_i), 2), "lamf_slider": round(float(lam_f), 2), "seed_input": int(seed), "text_input_area": text})

settings = Settings(users=int(users), notes=int(notes), share=int(share), gamma=round(float(gamma), 1), partisan=int(partisan), obs=int(obs), lam_i=round(float(lam_i), 2), lam_f=round(float(lam_f), 2), seed=int(seed), text=text)
with st.spinner("Rechne die Faktorisierung …"):
    a = _analysis(settings)
for message in a.errors:
    st.warning(message)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.metrics is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Rangkorrelation: Mittelwert", num(a.metrics["mittel"]["spearman"], 2), delta=f"{ev.pct(a.metrics['mittel']['partisan'])} parteiisch unter den besten {a.count}", delta_color="off")
    m2.metric("Rangkorrelation: Mindestlager", num(a.metrics["lager"]["spearman"], 2), delta=f"{ev.pct(a.metrics['lager']['partisan'])} parteiisch unter den besten {a.count}", delta_color="off")
    m3.metric("Rangkorrelation: brückenbasiert", num(a.metrics["mf"]["spearman"], 2), delta=f"{ev.pct(a.metrics['mf']['partisan'])} parteiisch unter den besten {a.count}", delta_color="off")
else:
    st.caption(f"Eigene Bewertungen: {a.users} Bewerter, {a.notes} Vorschläge; die wahre Qualität ist unbekannt, deshalb gibt es nur die Werte der Verfahren und ihre Reihenfolge.")
st.markdown("**Werte der Verfahren**" if a.metrics is None else "**Jeder Punkt ein Vorschlag: Wert des Verfahrens über der wahren Qualität**")
st.plotly_chart(build_scores(a), width="stretch", key="scores_chart")
if a.metrics is not None:
    st.caption("Ein dunkler Rand kennzeichnet die besten Vorschläge des Verfahrens. Das Ziel ist ein hoher Wert bei hoher Qualität und kein Rand bei parteiischen (roten oder blauen) Punkten niedriger Qualität.")
    st.table(ev.summary_rows(a))
    with st.expander("Die besten Vorschläge je Verfahren"):
        st.table(ev.ranking_rows(a))
else:
    st.table(ev.score_rows(a))
with st.expander("Bewerterfaktor und Zielfunktion der Faktorisierung"):
    st.markdown("**Bewerterfaktor nach Lager.** Trennen sich die Lager, hat die Faktorisierung die Lagerrichtung gefunden, ohne dass ihr jemand die Lager nannte.")
    st.plotly_chart(build_factor(a), width="stretch", key="factor_chart")
    st.caption(f"Die Faktorisierung brauchte {a.fit['steps']} Schritte" + (" und ist konvergiert." if a.fit["converged"] else " und hat die Toleranz nicht erreicht.") + f" Der Betrag der Korrelation zwischen Faktor und Lager ist {ev.pct(a.recovery)}.")
    st.plotly_chart(build_loss(a), width="stretch", key="loss_chart")
    st.caption("Zielfunktion (Fehlerquadrate plus Regularisierung) über die Schritte: jeder Schritt löst einen Block exakt, deshalb fällt sie in jedem Schritt.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wählt die Brücke die besseren Vorschläge als der Mittelwert?")
kind, verdict = ev.verdict_state(a)
getattr(st, kind)(verdict)

st.markdown("---")

# --- Reihen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Feste Reihen")
st.caption(
    f"Feste Programme, unabhängig von den Reglern: je Zeile {C.SERIES_RUNS} Zufallswelten mit festen Zufalls-Seeds ({C.DEFAULT_U} Bewerter, {C.DEFAULT_N} Vorschläge, {C.DEFAULT_PARTISAN} % parteiisch, {C.DEFAULT_OBS} % beobachtet); die Zeilen zeigen Mittelwerte über die Läufe, "
    "in jeder Reihe mit denselben Welten, nur der betrachtete Wert ändert sich. Jede Reihe rechnet auf Klick einige Sekunden."
)
tab1, tab2, tab3 = st.tabs(["Wirkt es nur bei ungleichen Lagern?", "Wie viel Regularisierung?", "Und bei schwacher Lagerneigung?"])
with tab1:
    st.markdown(f"**Lagergröße** bei Lagerneigung {num(C.DEFAULT_GAMMA, 1)}: je ungleicher die Lager, desto mehr zieht die mittlere Zustimmung zum großen Lager.")
    if st.button("Reihe rechnen", key="share_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe …"):
            _series_share()
        st.session_state["share_done"] = time.perf_counter() - t0
    if "share_done" in st.session_state:
        rows = _series_share()
        st.plotly_chart(build_share_series(rows), width="stretch", key="share_chart")
        st.table(ev.share_table(rows))
        kind, verdict = ev.verdict_share(rows)
        getattr(st, kind)(verdict)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['share_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab2:
    st.markdown(f"**Regularisierung des Intercepts** bei {C.SERIES_SHARE_MAIN} : {100 - C.SERIES_SHARE_MAIN}: ist der Intercept schwach regularisiert, nimmt er einen Teil der Lagerrichtung auf; mit starker Regularisierung bleibt sie im Faktor. Mittelwert und Mindestlager sind davon unberührt.")
    if st.button("Reihe rechnen", key="lam_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe …"):
            _series_lam()
        st.session_state["lam_done"] = time.perf_counter() - t0
    if "lam_done" in st.session_state:
        rows = _series_lam()
        st.plotly_chart(build_lam_series(rows), width="stretch", key="lam_chart")
        st.table(ev.lam_table(rows))
        kind, verdict = ev.verdict_lam(rows)
        getattr(st, kind)(verdict)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['lam_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab3:
    st.markdown(f"**Lagerneigung** bei {C.SERIES_SHARE_MAIN} : {100 - C.SERIES_SHARE_MAIN}: ohne Neigung sind die Lager für die Bewertung bedeutungslos, mit starker Neigung hängt die Zustimmung vor allem am Lager.")
    if st.button("Reihe rechnen", key="gamma_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe …"):
            _series_gamma()
        st.session_state["gamma_done"] = time.perf_counter() - t0
    if "gamma_done" in st.session_state:
        rows = _series_gamma()
        st.plotly_chart(build_gamma_series(rows), width="stretch", key="gamma_chart")
        st.table(ev.gamma_table(rows))
        kind, verdict = ev.verdict_gamma(rows)
        getattr(st, kind)(verdict)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['gamma_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt eine wahre Qualität** | Die Demo misst gegen eine bekannte Qualität; bei echten Vorschlägen ist sie unbekannt, und über „gut“ lässt sich streiten. | – |
| **Zwei Lager mit einer Richtung** | Echte Meinungen haben mehr als eine Achse; die Faktorisierung hat hier einen einzigen Faktor. | `nmf-demo` (mehrere Faktoren) |
| **Die Bewertungen sind ehrlich und unabhängig** | Koordiniertes Bewerten (Absprachen, viele Konten) kann das Ergebnis verschieben; die Demo modelliert es nicht. | – |
| **Genug Bewertungen aus beiden Lagern** | Bei wenigen oder fehlenden Bewertungen eines Lagers ist die Mindestzustimmung unsicher und der Faktor nicht bestimmt. | diese Demo (Beobachtungsanteil) |
| **Die Regularisierung ist richtig gewählt** | Eine schwache Regularisierung des Intercepts lässt das Verfahren kaum besser als den Mittelwert; die Wahl hängt von den Daten ab und braucht eine eigene Prüfung. | diese Demo (Reihe „Wie viel Regularisierung?“) |
| **Das Rauschen der Bewertung ist symmetrisch** | Systematische Verzerrungen (zum Beispiel, wer überhaupt bewertet) sind nicht modelliert. | – |
| **Modell, kein Befund über Plattformen** | Die Zahlen sagen, was die Rechnung für erzeugte oder eingegebene Bewertungen liefert, nicht, wie ein bestimmtes System arbeitet. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Bewertungen.** Bewerter $u$ gibt für Vorschlag $n$ die Bewertung $r_{un} \in \{0, 1\}$ ab, beobachtet für $(u, n) \in \Omega$. Erzeugt wird $r_{un} = 1$, wenn $\alpha q_n + \gamma\, \ell_n c_u + \varepsilon_{un} > \tau$ mit Qualität $q_n$, Neigung $\ell_n \in \{-1, 0, 1\}$, Lager $c_u \in \{-1, 1\}$ und $\varepsilon_{un} \sim \mathcal{N}(0, 1)$.

**Mittel und Mindestlager.** $\bar r_n = \frac{1}{|\Omega_n|} \sum_{u \in \Omega_n} r_{un}$ und $\min\bigl(\bar r_n^{A}, \bar r_n^{B}\bigr)$ mit den Mitteln der beiden Lager.

**Brückenbasierte Faktorisierung.**
$$\min \sum_{(u,n) \in \Omega} \bigl(r_{un} - \mu - i_u - i_n - f_u g_n\bigr)^2 + \lambda_i \Bigl(\sum_u i_u^2 + \sum_n i_n^2\Bigr) + \lambda_f \Bigl(\sum_u f_u^2 + \sum_n g_n^2\Bigr)$$
mit festem $\mu$ (Gesamtmittel). Gerankt wird nach $i_n$. Der Faktor $f_u g_n$ nimmt die Lagerrichtung auf; ist $\lambda_i \gg \lambda_f$, bleibt für den Intercept nur die Zustimmung, die sich nicht durch das Lager erklären lässt.

**Abwechselnde Ridge-Schritte.** Bei festen anderen Blöcken ist jeder Block eine Ridge-Regression mit geschlossener Lösung, zum Beispiel
$$i_n = \frac{\sum_{u \in \Omega_n} \bigl(r_{un} - \mu - i_u - f_u g_n\bigr)}{|\Omega_n| + \lambda_i}.$$
Deshalb fällt die Zielfunktion in jedem Schritt.

**Maße.** Rangkorrelation (Spearman) der Werte mit $q_n$; Anteil parteiischer Vorschläge ($\ell_n \neq 0$) unter den $k = 10$ besten; Korrelation des Bewerterfaktors $f_u$ mit dem Lager $c_u$ (Betrag, das Vorzeichen des Faktors ist beliebig).

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Wojcik u. a., „Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation“, 2022; die Dokumentation zu den Notizen der Gemeinschaft („Community Notes“) beschreibt eine Faktorisierung dieser Art.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
