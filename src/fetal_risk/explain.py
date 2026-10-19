from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import CLASS_NAMES
from .train import CVResult


def _class_shap(shap_vals, k: int) -> np.ndarray:
    """Handle both legacy (list of 2D) and modern (3D array) SHAP outputs."""
    if isinstance(shap_vals, list):
        return shap_vals[k]
    if shap_vals.ndim == 3:
        return shap_vals[:, :, k]
    return shap_vals


def oof_shap(cv: CVResult, target_class: int, scope: str = "all_folds"):
    """Compute SHAP values for one class.

    all_folds: every sample explained by the model that did not train on it
               (covers the full cohort, more stable than a single fold).
    last_fold: reproduces the original notebook behaviour.
    """
    folds = cv.folds if scope == "all_folds" else cv.folds[-1:]
    vals, raws = [], []
    for f in folds:
        sv = shap.TreeExplainer(f.model).shap_values(f.X_val_model)
        vals.append(_class_shap(sv, target_class))
        raws.append(f.X_val_raw)
    return np.vstack(vals), pd.concat(raws)


def plot_shap_summary(shap_values, X_raw, target_class: int, out_dir: Path,
                      max_display: int = 10, suffix: str = "") -> Path:
    name = CLASS_NAMES[target_class]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_raw, max_display=max_display, show=False)
    plt.title(f"Biomarker Attributions for {name} Fetal Risk (Class {target_class})",
              fontsize=13, pad=15)
    plt.tight_layout()
    path = out_dir / "figures" / f"shap_{name.lower()}{suffix}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
