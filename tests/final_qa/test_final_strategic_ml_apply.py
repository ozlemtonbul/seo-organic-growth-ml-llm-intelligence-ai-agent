from __future__ import annotations

from src.models.strategic_ml_only_router import HORIZONS


def test_required_ml_horizons():
    assert HORIZONS == (7, 14, 30, 90, 180, 365)
