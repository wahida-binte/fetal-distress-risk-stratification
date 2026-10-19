"""Smoke test on synthetic data so CI runs without the Kaggle CSV."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_risk.evaluate import compute_metrics  # noqa: E402
from fetal_risk.explain import oof_shap          # noqa: E402
from fetal_risk.train import run_cv              # noqa: E402

CFG = {
    "seed": 0,
    "cv": {"n_splits": 3, "shuffle": True},
    "preprocessing": {"standard_scale": True},
    "model": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3,
              "class_weight": "balanced"},
}


def _synthetic(n=300, p=6):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(np.digitize(X["f0"] + 0.3 * rng.normal(size=n), [-0.8, 0.9]))
    return X, y


def test_cv_shapes_and_metrics():
    X, y = _synthetic()
    cv = run_cv(X, y, CFG)
    assert cv.oof_probs.shape == (len(X), 3)
    assert np.allclose(cv.oof_probs.sum(axis=1), 1.0)
    m = compute_metrics(y, cv.oof_preds, cv.oof_probs)
    assert m["macro_ovr_roc_auc"] > 0.8


def test_oof_shap_covers_all_samples():
    X, y = _synthetic()
    cv = run_cv(X, y, CFG)
    sv, X_raw = oof_shap(cv, target_class=2, scope="all_folds")
    assert sv.shape == (len(X), X.shape[1])
    assert len(X_raw) == len(X)
