"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Sechs Bewerter (A, A, A, A, B, B), vier Vorschläge, vollständig bewertet. Vorschlag 1: A 1111, B 10 (Mittel 5/6, Mindestlager 1/2); 2: A 1110, B 11 (5/6, 3/4); 3: A 0001, B 00 (1/6, 0); 4: A 0000, B 11 (1/3, 0)."""
    camp = np.array([1, 1, 1, 1, -1, -1])
    ratings = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 0, 1]], dtype=float)
    return camp, ratings, np.ones_like(ratings, dtype=bool)
