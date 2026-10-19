from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def balanced_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    return {int(c): len(y) / (len(classes) * np.sum(y == c)) for c in classes}


@dataclass
class FoldArtifact:
    fold: int
    val_idx: np.ndarray
    model: lgb.LGBMClassifier
    X_val_model: pd.DataFrame   # what the model sees (scaled or raw)
    X_val_raw: pd.DataFrame     # raw values, used for SHAP colouring


@dataclass
class CVResult:
    oof_probs: np.ndarray
    oof_preds: np.ndarray
    folds: list[FoldArtifact] = field(default_factory=list)


def run_cv(X: pd.DataFrame, y: pd.Series, cfg: dict) -> CVResult:
    seed = cfg["seed"]
    n_classes = y.nunique()
    skf = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=cfg["cv"]["shuffle"], random_state=seed
    )

    oof_probs = np.zeros((len(X), n_classes))
    folds = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va = X.iloc[va]

        if cfg["preprocessing"]["standard_scale"]:
            scaler = StandardScaler().fit(X_tr)
            X_tr_m = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
            X_va_m = pd.DataFrame(scaler.transform(X_va), columns=X.columns, index=X_va.index)
        else:
            X_tr_m, X_va_m = X_tr, X_va

        m = cfg["model"]
        cw = balanced_class_weights(y_tr) if m.get("class_weight") == "balanced" else None
        model = lgb.LGBMClassifier(
            n_estimators=m["n_estimators"],
            learning_rate=m["learning_rate"],
            max_depth=m["max_depth"],
            class_weight=cw,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(X_tr_m, y_tr)
        oof_probs[va] = model.predict_proba(X_va_m)
        folds.append(FoldArtifact(fold, va, model, X_va_m, X_va))

    return CVResult(oof_probs, oof_probs.argmax(axis=1), folds)
