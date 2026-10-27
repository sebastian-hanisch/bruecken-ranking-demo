"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

U_MIN, U_MAX, DEFAULT_U = 40, 500, 300                    # Bewerter
N_MIN, N_MAX, DEFAULT_N = 10, 200, 100                    # Vorschläge
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 50, 95, 80          # Anteil des großen Lagers A in %
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 0.0, 4.0, 1.5       # Lagerneigung: wie stark das Lager die Zustimmung zu parteiischen Vorschlägen bestimmt
PARTISAN_MIN, PARTISAN_MAX, DEFAULT_PARTISAN = 0, 100, 60  # Anteil parteiischer Vorschläge in %
OBS_MIN, OBS_MAX, DEFAULT_OBS = 5, 100, 30               # Anteil der beobachteten Bewertungen in %
LAMI_MIN, LAMI_MAX, DEFAULT_LAMI = 0.05, 10.0, 5.0         # Regularisierung der Intercepts (Vorschläge und Bewerter)
LAMF_MIN, LAMF_MAX, DEFAULT_LAMF = 0.01, 1.0, 0.1          # Regularisierung der Faktoren
SEED_MAX, DEFAULT_SEED = 9999, 3
TEXT_MIN_RATERS, TEXT_MAX_RATERS = 3, 200                  # eigene Bewertungen: Zahl der Bewerter
TEXT_MIN_NOTES, TEXT_MAX_NOTES = 2, 30                     # … und Vorschläge

ALPHA = 1.0                                                # Gewicht der Qualität in der Zustimmung
TAU = 0.3                                                  # Schwelle der Zustimmung
TOPK = 10                                                  # so viele Vorschläge zählen als „oben“
MF_MAX_ITER, MF_TOL, MF_INIT_SD = 300, 1e-7, 1.0         # Faktorisierung: Schritte, Abbruch (relative Änderung der Zielfunktion), Streuung der Startwerte

RANKERS = {
    "mittel": "Mittlere Zustimmung",
    "lager": "Mindestzustimmung über beide Lager",
    "mf": "Brückenbasiert (Faktorisierung)",
}
RANKER_SHORT = {"mittel": "Mittelwert", "lager": "Mindestlager", "mf": "Brückenbasiert"}

COLOR_MEAN, COLOR_CAMP, COLOR_MF = "#4c78a8", "#7f7f7f", "#2F6B65"
COLOR_LEAN_A, COLOR_LEAN_B, COLOR_NEUTRAL = "#b04a3a", "#4c78a8", "#a0a0a0"

# Feste Reihen (gekoppelte Zufallswelten mit festen Zufalls-Seeds; Mittel über alle Läufe)
SERIES_SEED = 800000
SERIES_RUNS = 30
SERIES_SHARES = (50, 60, 70, 80, 90)
SERIES_LAMIS = (0.15, 0.5, 1.0, 2.0, 5.0, 10.0)
SERIES_GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SERIES_SHARE_MAIN = 80

NL = chr(10)

PRESETS = {
    "Ungleiche Lager (80 : 20)": {"users": 300, "notes": 100, "share": 80, "gamma": 1.5, "partisan": 60, "obs": 30, "lam_i": 5.0, "lam_f": 0.1, "seed": 3, "text": ""},
    "Gleich große Lager": {"users": 300, "notes": 100, "share": 50, "gamma": 1.5, "partisan": 60, "obs": 30, "lam_i": 5.0, "lam_f": 0.1, "seed": 3, "text": ""},
    "Schwache Regularisierung": {"users": 300, "notes": 100, "share": 80, "gamma": 1.5, "partisan": 60, "obs": 30, "lam_i": 0.15, "lam_f": 0.1, "seed": 3, "text": ""},
    "Keine Lagerneigung": {"users": 300, "notes": 100, "share": 80, "gamma": 0.0, "partisan": 60, "obs": 30, "lam_i": 5.0, "lam_f": 0.1, "seed": 3, "text": ""},
    "Starke Polarisierung": {"users": 300, "notes": 100, "share": 80, "gamma": 3.0, "partisan": 60, "obs": 30, "lam_i": 5.0, "lam_f": 0.1, "seed": 3, "text": ""},
    "Handbeispiel (sechs Bewerter)": {"users": 300, "notes": 100, "share": 80, "gamma": 1.5, "partisan": 60, "obs": 30, "lam_i": 5.0, "lam_f": 0.1, "seed": 3,
                                     "text": "A 1 1 0 0" + NL + "A 1 1 0 0" + NL + "A 1 1 0 0" + NL + "A 1 0 1 0" + NL + "B 1 1 0 1" + NL + "B 0 1 0 1"},
}
PRESET_HELP = {
    "Ungleiche Lager (80 : 20)": "Voreinstellung: 300 Bewerter, 80 % im großen Lager, Lagerneigung 1,5. Rangkorrelation zur wahren Qualität: Mittelwert 0,75, Mindestlager 0,74, brückenbasiert 0,94; unter den besten zehn sind beim Mittelwert 50 %, brückenbasiert 30 % parteiische Vorschläge.",
    "Gleich große Lager": "Gleich große Lager (50 : 50): der Mittelwert reicht fast (0,93 gegen 0,95 brückenbasiert); das Mindestlager ist mit 0,77 schlechter (mehr Bewerter ändern daran in der Simulation nichts; vermutete Ursache: das Minimum ordnet parteiische Vorschläge allein nach dem fremden Lager).",
    "Schwache Regularisierung": "Dieselbe Welt wie die Voreinstellung, aber λ = 0,15 statt 5: das brückenbasierte Verfahren fällt von 0,94 auf 0,78 (50 % parteiisch), kaum über dem Mittelwert mit 0,75.",
    "Keine Lagerneigung": "Lagerneigung 0: das Lager spielt für die Bewertung keine Rolle, alle drei Verfahren liegen bei 0,97 bis 0,98, und der Bewerterfaktor folgt dem Lager zu 0 %.",
    "Starke Polarisierung": "Lagerneigung 3,0: Mittelwert 0,51, Mindestlager 0,35, brückenbasiert 0,70; unter den besten zehn sind beim Mittelwert 50 %, brückenbasiert 0 % parteiische Vorschläge.",
    "Handbeispiel (sechs Bewerter)": "Von Hand: vier Bewerter in Lager A, zwei in B, vier Vorschläge. Vorschlag 1 (A 1111, B 10) und Vorschlag 2 (A 1110, B 11) haben beide die mittlere Zustimmung 5/6; die Mindestzustimmung unterscheidet sie (1/2 gegen 3/4) und stellt Vorschlag 2 vorn. Bei so wenigen Bewertungen hat die Faktorisierung kaum Spielraum und stellt Vorschlag 1 vorn.",
}
