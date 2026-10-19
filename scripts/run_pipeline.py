"""End-to-end pipeline: load data, stratified CV, evaluate, explain.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_risk.data import load_dataset            # noqa: E402
from fetal_risk.evaluate import (                   # noqa: E402
    compute_metrics, plot_confusion_matrix, print_report, save_metrics,
)
from fetal_risk.explain import oof_shap, plot_shap_summary  # noqa: E402
from fetal_risk.train import run_cv                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", help="Override data.path from the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data:
        cfg["data"]["path"] = args.data

    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(**cfg["data"])
    print(f"Cohort: {X.shape[0]} recordings, {X.shape[1]} CTG features")
    print("Class distribution:\n", y.value_counts(normalize=True).sort_index(), "\n")

    cv = run_cv(X, y, cfg)

    metrics = compute_metrics(y, cv.oof_preds, cv.oof_probs)
    print_report(y, cv.oof_preds, metrics)
    print("Saved:", save_metrics(metrics, out_dir))
    print("Saved:", plot_confusion_matrix(y, cv.oof_preds, out_dir))

    e = cfg["explain"]
    sv, X_raw = oof_shap(cv, e["target_class"], e["scope"])
    print("Saved:", plot_shap_summary(sv, X_raw, e["target_class"], out_dir, e["max_display"]))


if __name__ == "__main__":
    main()
