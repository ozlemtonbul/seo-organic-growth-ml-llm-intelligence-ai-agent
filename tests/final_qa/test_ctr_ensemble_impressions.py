from __future__ import annotations

import numpy as np

from src.models.ctr_ensemble_impressions import _fold_starts


def test_33_samples_have_multiple_oof_folds():
    starts = _fold_starts(33)
    assert len(starts) >= 2
    assert all(isinstance(v, int) for v in starts)
