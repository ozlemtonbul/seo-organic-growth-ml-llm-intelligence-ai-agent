from __future__ import annotations

from src.models.ml_only_ctr_ensemble import (
    ML_CANDIDATES,
    _fold_starts,
)


def test_only_ml_members_are_defined():
    assert set(ML_CANDIDATES) == {
        "CTRRidge",
        "CTRGradientBoosting",
        "CTRRandomForest",
        "CTRHGBR",
    }


def test_33_samples_produce_multiple_folds():
    assert len(_fold_starts(33)) >= 2
