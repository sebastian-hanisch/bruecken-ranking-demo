# Brückenbasiertes Ranking: Zustimmung über Lager hinweg – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-bruecken-ranking-demo.streamlit.app/)** -->

Gelber Kandidat der Linie **Abstimmen und Fair Teilen** (Konzepte-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“). Verwandt: [nmf-demo](https://github.com/sebastian-hanisch/nmf-demo) (Matrixfaktorisierung), [meinungsbildung-demo](https://github.com/sebastian-hanisch/meinungsbildung-demo) (Meinungsbildung im Netz) und [quadratisches-waehlen-demo](https://github.com/sebastian-hanisch/quadratisches-waehlen-demo) (Mehrheit und Intensität).
Die Demo fragt: **Eine Plattform lässt Vorschläge von Nutzern als hilfreich oder nicht hilfreich bewerten und zeigt die besten zuerst. Gehören die Bewerter zwei Lagern an, belohnt die mittlere Zustimmung, was das größere Lager mag. Wie gut ordnet ein brückenbasiertes Verfahren, das die Lagerrichtung herausrechnet, die Vorschläge nach ihrer wahren Qualität, und wovon hängt das ab?** Verglichen werden die **mittlere Zustimmung**, die **Mindestzustimmung über beide Lager** (braucht die Lagerangabe jedes Bewerters) und das **brückenbasierte Verfahren** (Faktorisierung, die Lager nicht kennt), gegen die in der Simulation bekannte Qualität.

Der Kern ist eigener Code (`numpy`, abwechselnde Ridge-Schritte in Matrix-Vektor-Form). Unabhängige Orakel in den Tests: die **Zielfunktion gegen eine Referenz und gegen Schleifen**, **jeder Lösungsschritt gegen eine zweite Umsetzung in reinen Python-Schleifen** und gegen die **Normalgleichung** der Ridge-Regression, ein **stationärer Punkt per Gradient**, die **Rangkorrelation gegen `scipy`**, **Mittel und Mindestzustimmung gegen Schleifen** und ein **rauschfreier Rang-1-Fall**.

## Warum dieses Problem

Die mittlere Zustimmung ordnet gut, solange es keine Lager gibt oder beide gleich groß sind. Ist ein Lager **größer** und hängt die Zustimmung zu parteiischen Vorschlägen am Lager, steigen die Vorschläge des großen Lagers nach oben: bei 80 : 20 liegt die Rangkorrelation des Mittelwerts mit der wahren Qualität bei **0,76**, und **59 %** der besten zehn Vorschläge sind parteiisch (bei gleich großen Lagern 0,94 und 34 %). Das **brückenbasierte Verfahren** zerlegt die Bewertungen in einen **Lagerfaktor** und einen **Intercept je Vorschlag**; gerankt wird nach dem Intercept. Es erreicht bei 80 : 20 **0,94** und **36 %** (die Reihen). Das hängt an der **Regularisierung**: ist der Intercept nur schwach regularisiert, nimmt er einen Teil der Lagerrichtung auf, und das Verfahren liegt kaum über dem Mittelwert (0,80 bei λ = 0,15, gegen 0,94 bei λ = 10).

## Modell

**Welt.** U Bewerter in Lager A (Anteil „Größe des großen Lagers“) oder B; N Vorschläge mit wahrer Qualität q_n (standardnormalverteilt) und Lagerneigung ℓ_n ∈ {−1, 0, +1} (Anteil „parteiisch“ je zur Hälfte A und B). Ein Bewerter findet Vorschlag n hilfreich (r = 1), wenn q_n + γ · ℓ_n · Lager + Rauschen > 0,3 mit γ = „Lagerneigung“; nur ein Anteil der Bewertungen wird beobachtet. Eigene Bewertungen lassen sich als Zeilen mit Lager und 1/0/- eingeben (dann ist die Qualität unbekannt).
**Mittlere Zustimmung.** Anteil „hilfreich“ unter den beobachteten Bewertungen eines Vorschlags; ohne Bewertung gilt das Gesamtmittel.
**Mindestzustimmung.** Das Minimum der mittleren Zustimmungen der beiden Lager (ein Lager ohne Bewertung eines Vorschlags: Mittel aller Bewerter).
**Brückenbasiert.** r ≈ μ + i_u + i_n + f_u · g_n mit festem μ (Gesamtmittel); minimiert wird die Summe der Fehlerquadrate über die beobachteten Bewertungen plus λ_i · (Σ i_u² + Σ i_n²) + λ_f · (Σ f_u² + Σ g_n²). Gelöst wird durch **abwechselnde Ridge-Schritte** (je Block eine exakte Lösung, die Zielfunktion fällt also in jedem Schritt), Startwerte der Faktoren normalverteilt mit Streuung 1, Abbruch bei relativer Änderung der Zielfunktion unter 10⁻⁷ (höchstens 300 Schritte). Gerankt wird nach i_n; die Lager bekommt das Verfahren nicht genannt.
**Maße.** Rangkorrelation (Spearman) der Werte mit q_n; Anteil parteiischer Vorschläge unter den besten zehn; mittlere Qualität der besten zehn; Betrag der Korrelation des Bewerterfaktors f_u mit dem Lager. Formeln und Lesehilfe stehen im 📐-Expander der App.

## Methodik

- **Kern** (`brk_algorithm.py`): Ränge und Rangkorrelation, mittlere und Mindestzustimmung, Faktorisierung (ALS mit schneller Zielfunktion), Rangliste, Lagererkennung.
- **Szenario und Auswertung** (`brk_scenario.py`, `brk_evaluation.py`): Welten aus dem Zufalls-Seed, Texteingabe mit Prüfung, Tabellen, drei feste Reihen, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): sechs Schnellstart-Szenarien (eines von Hand), Regler für Bewerter, Vorschläge, Lagergröße, Lagerneigung, Parteilichkeit, Beobachtungsanteil, beide Regularisierungen und Zufalls-Seed, Textfeld für eigene Bewertungen, Werte je Verfahren über der wahren Qualität, Bewerterfaktor und Zielfunktion, Kernfrage, drei Reiter mit festen Reihen, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Die Reihen laufen auf **30 Zufallswelten** je Zeile (300 Bewerter, 100 Vorschläge, 60 % parteiisch, 30 % beobachtet, Faktor-Regularisierung 0,1); die Zeilen zeigen Mittelwerte über die Läufe, **in jeder Reihe mit denselben Welten** (nur der betrachtete Wert ändert sich). Die Rangkorrelation ist die Spearman-Korrelation der Werte mit der wahren Qualität.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Die schnelle Zielfunktion gleicht der **elementweisen Referenz und einer Schleifenrechnung** (10 Zufallsprobleme, beliebige Parameter); jeder Lösungsschritt gleicht einer **zweiten Umsetzung in Schleifen** (6 Probleme, 7 Schritte, 10⁻¹⁰); der Faktor eines Vorschlags ist die Lösung der **Normalgleichung**; am konvergierten Punkt verschwindet der **Gradient** (Betrag unter 10⁻⁴ je Koordinate); die Rangkorrelation gleicht **scipy** (40 Fälle mit Gleichständen); exakt zerlegbare Bewertungen werden bis auf 10⁻³ reproduziert. Die Zielfunktion fällt in jedem Schritt. | `test_oracle_bruecken_ranking.py`, `test_algorithm.py::test_zielfunktion_fällt_in_jedem_schritt` |
| Von Hand: sechs Bewerter | Vier Bewerter in Lager A, zwei in B, vier Vorschläge, vollständig bewertet: Vorschlag 1 (A 1111, B 10) und Vorschlag 2 (A 1110, B 11) haben beide die **mittlere Zustimmung 5/6**; die **Mindestzustimmung** unterscheidet sie: **1/2 gegen 3/4**, Vorschlag 2 steht vorn. Die Faktorisierung stellt bei so wenigen Bewertungen Vorschlag 1 vorn (Intercepts 0,19 und 0,07): ein Beleg dafür, dass sie Daten braucht. | `test_algorithm.py::test_mittlere_zustimmung_handbeispiel`, `…::test_zustimmung_je_lager_und_mindestzustimmung`, `test_evaluation.py::test_handbeispiel_exakte_werte` |
| Wirkt es nur bei ungleichen Lagern? | Rangkorrelation (Mittelwert / Mindestlager / brückenbasiert) bei 50 : 50 **0,94 / 0,78 / 0,95**, 60 : 40 0,90 / 0,78 / 0,95, 70 : 30 0,82 / 0,77 / 0,94, **80 : 20 0,76 / 0,77 / 0,94**, **90 : 10 0,69 / 0,74 / 0,92**. Parteiische Vorschläge unter den besten zehn: Mittelwert **34 → 76 %**, brückenbasiert 33 → 37 %, Mindestlager 17 bis 19 %. Der Bewerterfaktor folgt dem Lager zu 94 bis 98 %. | `test_claims.py::test_bei_gleich_großen_lagern_genügt_der_mittelwert`, `…::test_mit_ungleichen_lagern_verzerrt_der_mittelwert_und_die_brücke_hilft`, `…::test_mindestlager_senkt_parteiisch_am_stärksten_verliert_aber_rangtreue`, `…::test_der_bewerterfaktor_findet_die_lager` |
| Wie viel Regularisierung? | Bei 80 : 20 mit λ des Intercepts 0,15 / 0,5 / 1 / 2 / 5 / 10: brückenbasiert **0,80 / 0,84 / 0,88 / 0,91 / 0,93 / 0,94**, parteiisch unter den besten zehn **50 / 41 / 34 / 30 / 28 / 28 %**; der Mittelwert bleibt bei 0,77 und 57 %. Alle Anpassungen konvergieren (30 von 30). | `test_claims.py::test_schwache_regularisierung_lässt_die_brücke_beim_mittelwert`, `…::test_starke_regularisierung_bringt_die_brücke_zum_ziel` |
| Und bei schwacher Lagerneigung? | Bei 80 : 20 mit Lagerneigung 0 / 0,5 / 1,0 / 1,5 / 2,0 / 3,0: Mittelwert **0,99 / 0,96 / 0,87 / 0,76 / 0,67 / 0,50**, brückenbasiert **0,98 / 0,97 / 0,96 / 0,93 / 0,88 / 0,69**, Mindestlager 0,97 / 0,95 / 0,87 / 0,76 / 0,65 / 0,42. Ohne Neigung folgt der Bewerterfaktor dem Lager zu **4 %** und ist nicht nötig; ab Neigung 1,5 zu mindestens 97 %. | `test_claims.py::test_ohne_lagerneigung_sind_alle_verfahren_gleich_gut_und_der_faktor_findet_nichts`, `…::test_mit_wachsender_neigung_sinkt_der_mittelwert_und_die_brücke_bleibt_vorn` |
| Schnellstart-Szenarien | Voreinstellung (80 : 20, Seed 3): Rangkorrelation 0,75 / 0,74 / 0,94, parteiisch 50 / 20 / 30 %. Gleich große Lager: 0,93 / 0,77 / 0,95. Schwache Regularisierung (λ = 0,15): brückenbasiert 0,78. Keine Lagerneigung: 0,98 / 0,97 / 0,98. Starke Polarisierung (3,0): 0,51 / 0,35 / 0,70, parteiisch 50 / 0 / 0 %. | `test_evaluation.py::test_hilfetext_des_presets_stimmt_mit_der_rechnung` |
| Vormessung | Eine Vormessung vor dem Bau (Wegwerfskript, 60 bis 100 Wiederholungen) zeigte denselben Sinn: bei gleich großen Lagern genügt der Mittelwert, bei 80 : 20 hilft die Faktorisierung nur mit starker Regularisierung. **Keine Zahl daraus** steht hier, alle Zahlen sind die der festen Reihen. | – |

## Ehrliche Grenzen

- **Die wahre Qualität ist nur in der Simulation bekannt.** Bei echten Vorschlägen ist sie unbekannt, und über „gut“ lässt sich streiten; die Demo zeigt, was die Verfahren bei **bekannter** Qualität leisten.
- **Zwei Lager mit einer Richtung, ein Faktor.** Echte Meinungen haben mehr Achsen; mehrere Faktoren sind nicht umgesetzt.
- **Koordiniertes Bewerten** (Absprachen, viele Konten) kann jedes Verfahren verschieben; nicht modelliert.
- **Genug Bewertungen aus beiden Lagern.** Bei wenigen oder fehlenden Bewertungen ist die Mindestzustimmung unsicher und der Faktor nicht bestimmt (siehe Handbeispiel). Die Beobachtungsdichte ist als Regler da, aber nicht als Reihe gemessen.
- **Die Regularisierung ist eine Wahl.** Die Reihe zeigt, dass sie über Erfolg entscheidet; wie man sie bei echten Daten findet (zum Beispiel durch Kreuzvalidierung), ist nicht gemessen.
- **Die Mindestzustimmung** drückt parteiische Vorschläge am stärksten, verliert aber Rangtreue, weil die Mittel des kleinen Lagers rauschen; sie braucht außerdem die Lagerangabe.
- **Symmetrisches Rauschen.** Systematische Verzerrungen (wer überhaupt bewertet) sind nicht modelliert.
- **Reihen mit 30 Welten je Zeile** und festen Zufalls-Seeds; Unterschiede von wenigen Hundertsteln liegen im Rahmen der Stichprobe.
- **Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Wojcik u. a., „Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce Misinformation“, 2022; die Dokumentation zu den Notizen der Gemeinschaft („Community Notes“) beschreibt eine Faktorisierung dieser Art.

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9202, Präfix `brk_`.
- **Plan:** „Matrixfaktorisierung auf Bewerter × Vorschlag, Vorschläge zählen nur bei Zustimmung über Lager“. Umgesetzt mit bekannter Qualität als Maßstab, drei Verfahren und drei Reihen; die Kulisse „Verbesserungsvorschläge in zwei Belegschaftslagern“ ist hier die Simulation mit zwei Lagern.
- **Die Vormessung lief mit schwach regularisierten Faktoren und kleinen Startwerten** (λ = 0,15 für den Intercept, 0,03 für den Faktor) und zeigte, dass das Verfahren dann beim Mittelwert bleibt; erst die starke Regularisierung (λ = 5, 10) und größere Startwerte machen es besser. Deshalb ist die Regularisierung ein Regler und eine eigene Reihe, und die Voreinstellung ist λ = 5.
- **Abbruchtoleranz:** 10⁻⁷ statt 10⁻¹⁰ (relative Änderung der Zielfunktion): die Rangkorrelation ändert sich in den ersten drei Stellen nicht (0,933 bei beiden), die Rechenzeit sinkt auf etwa die Hälfte der Schritte.
- **Rechenzeit:** die elementweise Formulierung brauchte etwa 0,2 bis 1 s je Anpassung, die Matrix-Vektor-Form mit ausmultiplizierter Zielfunktion etwa 0,02 s; die Reihen laufen dadurch in je 4 bis 5 s. Die ausmultiplizierte Zielfunktion ist im Test gegen die elementweise Referenz und eine Schleifenrechnung geprüft.
- **Reglerspanne:** Lagerneigung 0 bis 4 (die Reihen bis 3; 3 ist nicht die Obergrenze, damit „Starke Polarisierung“ nicht am Rand liegt), Bewerter 40 bis 500, Vorschläge 10 bis 200, großes Lager 50 bis 95 %, beobachtet 5 bis 100 %.

## Verifikation

- **Umfang:** 190 Tests.
- **Kern:** jede Einheit an Mini-Fällen von Hand (Ränge mit Gleichstand, Rangkorrelation 0,8, Mittel und Mindestzustimmung des Handbeispiels, ein von Hand gerechneter Schritt mit geschrumpften Intercepts, Abbruch und Höchstzahl der Schritte).
- **Orakel:** Zielfunktion, Lösungsschritte in Schleifen, Normalgleichung, Gradient, scipy, Rang-1-Fall (siehe Befunde).
- **Reihen und Texte:** jede Zahl dieser README in `tests/test_claims.py` (Richtungen und Bänder, nicht die letzte Stelle: die Zufallsströme von numpy können sich zwischen Versionen unterscheiden) und `tests/test_evaluation.py` (Hilfetexte der Presets, Verdikte und Tabellen als reine Funktionen), Randfälle und Adresszeile in `tests/test_edge_cases.py`.
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich, Schnellstart oben, aktives Preset hervorgehoben, AppTest jeder Button und jeder Regler an den Rändern, Footer, Laufzeit-Importe).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 117 Stellen in Kern und Auswertung (`brk_algorithm.py`, `brk_evaluation.py`), **115 gefangen** (3 davon über wortgleiche Zeilen in Nachbar-Demos), **2 als äquivalent festgehalten** (in `average_ranks` ändert die Schwelle „mehr als ein Eintrag gleich“ nichts, weil der Mittelwert eines einzelnen Rangs dieser Rang ist), keine Zeitüberschreitung. Der erste Lauf ließ 3 überleben; **eine echte Lücke** war die Grenze „Regularisierung genau null ist ungültig“ für den Faktor (ein neuer Test). Zuvor waren schon beim Schreiben der Tests die Schwellen der Verdikte an handgesetzten Zeilen festgelegt (Unterschiede knapp unter und über 0,02 und 0,05).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `brk_algorithm.py` | Kern: Ränge, Mittel und Mindestzustimmung, Faktorisierung (ALS), Rangliste, Lagererkennung |
| `brk_constants.py` | Regler-Grenzen, feste Reihen, Farben, Presets |
| `brk_evaluation.py` | Analyse, Tabellen, Verdikte, drei feste Reihen |
| `brk_layout.py` | Fester Kopf (Stylesheet) |
| `brk_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `brk_scenario.py` | Welt, Bewertungen, Texteingabe |
| `brk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Orakel, Reihen und Claims, Auswertung und Presets, Randfälle, Szenario, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Faktoren** und **mehr als zwei Lager**: eine eigene Rechnung mit Wahl der Faktorzahl.
- **Wahl der Regularisierung aus den Daten** (Kreuzvalidierung): die Demo zeigt nur, wie viel an ihr hängt.
- **Manipulation** (koordiniertes Bewerten) und **ungleiche Beteiligung** der Lager an der Bewertung.
- **Echte Bewertungsdaten**: keine Quelle.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy (SciPy nur in den Tests).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
