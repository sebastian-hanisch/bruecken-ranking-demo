"""Szenario: Bewerter in zwei Lagern, Vorschläge mit Qualität und Lagerneigung, beobachtete Bewertungen (erzeugt oder eingegeben). Deterministisch aus dem Zufalls-Seed."""

from dataclasses import dataclass

import numpy as np

import brk_constants as C


@dataclass
class World:
    camp: np.ndarray               # +1 (Lager A) oder −1 (Lager B) je Bewerter
    ratings: np.ndarray            # 1 = hilfreich, 0 = nicht hilfreich (Bewerter × Vorschläge)
    mask: np.ndarray               # True: Bewertung beobachtet
    quality: np.ndarray | None     # wahre Qualität je Vorschlag (nur bei erzeugten Welten)
    lean: np.ndarray | None        # Lagerneigung je Vorschlag: +1 (A), −1 (B), 0 (neutral)


def make_world(users: int, notes: int, share: int, gamma: float, partisan: int, obs: int, seed: int) -> World:
    """Ein Bewerter findet einen Vorschlag hilfreich, wenn ALPHA · Qualität + gamma · Neigung · Lager + Rauschen > TAU. Beobachtet wird jede Bewertung unabhängig mit Wahrscheinlichkeit obs %."""
    if users < 1 or notes < 1 or not 0 <= share <= 100 or gamma < 0 or not 0 <= partisan <= 100 or not 0 < obs <= 100:
        raise ValueError("Bewerter, Vorschläge, Lagergröße, Neigung, Parteilichkeit und Beobachtungsanteil passen nicht")
    rng = np.random.default_rng(seed)
    camp = np.where(rng.random(users) < share / 100, 1, -1)
    quality = rng.normal(size=notes)
    lean = np.where(rng.random(notes) < partisan / 100, rng.choice([-1, 1], size=notes), 0)
    utility = C.ALPHA * quality[None, :] + gamma * lean[None, :] * camp[:, None] + rng.normal(size=(users, notes))
    mask = rng.random((users, notes)) < obs / 100
    return World(camp=camp, ratings=(utility > C.TAU).astype(float), mask=mask, quality=quality, lean=lean)


def parse_ratings(text: str):
    """Eingabe: eine Zeile je Bewerter, zuerst der Buchstabe des Lagers (A oder B), dann je Vorschlag 1 (hilfreich), 0 (nicht hilfreich) oder - (nicht bewertet), getrennt durch Leerzeichen oder Komma.
    Rückgabe (World oder None, Fehlermeldung oder None); leerer Text: (None, None)."""
    text = (text or "").strip()
    if not text:
        return None, None
    camps, rows = [], []
    for line in text.splitlines():
        parts = [t for t in line.replace(",", " ").split() if t]
        if not parts:
            continue
        head = parts[0].upper()
        if head not in ("A", "B"):
            return None, f"Jede Zeile beginnt mit dem Lager A oder B (gefunden: „{parts[0]}“)."
        row = parts[1:]
        if any(t not in ("0", "1", "-") for t in row):
            bad = next(t for t in row if t not in ("0", "1", "-"))
            return None, f"„{bad}“ ist keine Bewertung (erlaubt: 1, 0 oder - für nicht bewertet)."
        camps.append(1 if head == "A" else -1)
        rows.append(row)
    if len({len(r) for r in rows}) != 1:
        return None, "Jede Zeile braucht gleich viele Bewertungen (eine Spalte je Vorschlag)."
    users, notes = len(rows), len(rows[0])
    if not C.TEXT_MIN_RATERS <= users <= C.TEXT_MAX_RATERS:
        return None, f"Es müssen {C.TEXT_MIN_RATERS} bis {C.TEXT_MAX_RATERS} Bewerter sein (eingegeben: {users})."
    if not C.TEXT_MIN_NOTES <= notes <= C.TEXT_MAX_NOTES:
        return None, f"Es müssen {C.TEXT_MIN_NOTES} bis {C.TEXT_MAX_NOTES} Vorschläge sein (eingegeben: {notes})."
    mask = np.array([[t != "-" for t in r] for r in rows])
    ratings = np.array([[1.0 if t == "1" else 0.0 for t in r] for r in rows])
    if not mask.any():
        return None, "Mindestens eine Bewertung muss vorhanden sein."
    return World(camp=np.array(camps), ratings=ratings, mask=mask, quality=None, lean=None), None
