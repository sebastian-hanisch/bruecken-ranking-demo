"""Randfälle: Extremwerte der Regler, Invarianten jeder Analyse, Adresszeile und eigene Bewertungen."""

import itertools
import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import brk_constants as C
import brk_evaluation as ev

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
HAND = C.PRESETS["Handbeispiel (sechs Bewerter)"]["text"]
GRID = list(itertools.product((C.U_MIN, 120), (C.N_MIN, 60), (C.SHARE_MIN, C.SHARE_MAX), (C.GAMMA_MIN, C.GAMMA_MAX), (C.OBS_MIN, C.OBS_MAX), (C.LAMI_MIN, C.LAMI_MAX)))


@pytest.mark.parametrize("users,notes,share,gamma,obs,lam_i", GRID)
def test_invarianten_an_den_grenzen(users, notes, share, gamma, obs, lam_i):
    a = ev.analyse(ev.Settings(users=users, notes=notes, share=share, gamma=gamma, partisan=C.PARTISAN_MAX, obs=obs, lam_i=lam_i, lam_f=C.LAMF_MIN, seed=3))
    assert (a.users, a.notes) == (users, notes) and a.errors == [] and a.world.ratings.shape == (users, notes)
    for k in ("mittel", "lager", "mf"):
        assert np.isfinite(a.scores[k]).all() and len(a.tops[k]) == min(C.TOPK, notes)
        assert -1 <= a.metrics[k]["spearman"] <= 1 and 0 <= a.metrics[k]["partisan"] <= 1
    assert np.isfinite(a.fit["losses"]).all() and all(b <= x + 1e-6 for x, b in zip(a.fit["losses"], a.fit["losses"][1:]))
    assert 0 <= a.recovery <= 1 and ev.verdict_state(a)[1]


def test_nur_ein_lager_vertreten():
    a = ev.analyse(ev.Settings(text="A 1 0 1\nA 0 1 1\nA 1 1 0"))
    assert a.source == "eingegeben" and np.isfinite(a.scores["lager"]).all()
    assert a.scores["lager"] == pytest.approx(np.minimum(a.scores["mittel"], a.scores["mittel"]))                      # Lager B fehlt: seine Zustimmung ist das Gesamtmittel
    assert a.recovery == 0.0


def test_ein_vorschlag_ohne_bewertung_bekommt_das_gesamtmittel():
    a = ev.analyse(ev.Settings(text="A 1 0 -\nB 0 1 -\nA 1 1 -"))
    assert a.scores["mittel"][2] == pytest.approx(4 / 6) and np.isfinite(a.scores["mf"]).all()


def test_alle_bewertungen_gleich_gibt_gleichstand_ohne_fehler():
    a = ev.analyse(ev.Settings(text="A 1 1 1\nB 1 1 1\nA 1 1 1"))
    assert np.allclose(a.scores["mittel"], 1.0) and np.allclose(a.scores["mf"], a.scores["mf"][0])
    assert [int(i) for i in a.tops["mittel"]] == [0, 1, 2]                                                           # Gleichstand: kleinere Nummer zuerst


# --- App: Adresszeile und eigene Bewertungen ---------------------------------------------------------------------------------------------------


def _run(**params):
    at = AppTest.from_file(APP, default_timeout=180)
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_permalink_mit_eigenen_bewertungen_füllt_das_feld_und_sperrt_den_erzeuger():
    at = _run(text=HAND)
    assert at.sidebar.text_area(key="text_input_area").value == HAND
    for key in ("users_slider", "notes_slider", "share_slider", "gamma_slider", "partisan_slider", "obs_slider"):
        assert at.sidebar.slider(key=key).proto.disabled
    assert not at.sidebar.slider(key="lami_slider").proto.disabled
    assert not at.metric and any("Eigene Bewertungen: 6 Bewerter, 4 Vorschläge" in c.value for c in at.caption)


def test_permalink_werte_werden_begrenzt_und_schlechte_ignoriert():
    at = _run(users="9999", notes="x", share="-5", gamma="9.5", partisan="150", obs="0", lami="nan", lamf="0", seed="-3")
    assert at.sidebar.slider(key="users_slider").value == C.U_MAX and at.sidebar.slider(key="notes_slider").value == C.DEFAULT_N
    assert at.sidebar.slider(key="share_slider").value == C.SHARE_MIN and at.sidebar.slider(key="gamma_slider").value == C.GAMMA_MAX
    assert at.sidebar.slider(key="partisan_slider").value == C.PARTISAN_MAX and at.sidebar.slider(key="obs_slider").value == C.OBS_MIN
    assert at.sidebar.slider(key="lami_slider").value == C.LAMI_MIN and at.sidebar.slider(key="lamf_slider").value == C.LAMF_MIN and at.sidebar.number_input(key="seed_input").value == 0


def test_zu_langer_text_aus_der_adresszeile_wird_abgeschnitten():
    at = _run(text="A 1 0\n" * 5000)
    assert len(at.sidebar.text_area(key="text_input_area").value) <= 4000


def test_schlechte_eingabe_zeigt_eine_warnung_und_die_app_läuft_weiter():
    at = _run(text="A 1 0\nC 0 1\nA 1 1")
    assert any("Eigene Bewertungen" in w.value for w in at.warning) and not at.sidebar.slider(key="users_slider").proto.disabled


def test_standardlauf_zeigt_drei_kennzahlen_und_ein_urteil():
    at = _run()
    assert [m.label for m in at.metric] == ["Rangkorrelation: Mittelwert", "Rangkorrelation: Mindestlager", "Rangkorrelation: brückenbasiert"]
    assert at.metric[2].value == "0,94" and any("Rangkorrelation zur Qualität: Mittelwert 0,75, brückenbasiert 0,94" in s.value for s in at.success)


def test_regler_ändern_hält_die_app_am_laufen():
    at = _run()
    for key, value in (("users_slider", 40), ("notes_slider", 10), ("share_slider", 95), ("gamma_slider", 3.0), ("partisan_slider", 0), ("obs_slider", 5), ("lami_slider", 0.05), ("lamf_slider", 1.0)):
        at.sidebar.slider(key=key).set_value(value).run()
        assert not at.exception


def test_die_reihen_laufen_auf_klick():
    at = _run()
    for key in ("share_button", "lam_button", "gamma_button"):
        at.button(key=key).click().run(timeout=180)
        assert not at.exception, [str(e) for e in at.exception]
    assert len(at.table) == 5                                                       # Zusammenfassung, Rangliste (im Expander) und drei Reihentabellen
