import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from . import CLASS_NAMES


def compute_metrics(y, oof_preds, oof_probs) -> dict:
    report = classification_report(
        y, oof_preds, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    report["macro_ovr_roc_auc"] = float(roc_auc_score(y, oof_probs, multi_class="ovr"))
    return report


def print_report(y, oof_preds, metrics: dict) -> None:
    print("=" * 60)
    print("     OUT-OF-FOLD MULTI-CLASS CLINICAL EVALUATION")
    print("=" * 60)
    print(classification_report(y, oof_preds, target_names=CLASS_NAMES, digits=4))
    print(f"Macro One-vs-Rest ROC-AUC: {metrics['macro_ovr_roc_auc']:.4f}")


def save_metrics(metrics: dict, out_dir: Path) -> Path:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def plot_confusion_matrix(y, oof_preds, out_dir: Path) -> Path:
    cm = confusion_matrix(y, oof_preds, normalize="true")
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Normalized Confusion Matrix (Clinical Recall)", fontsize=12)
    plt.xlabel("Predicted Risk Tier")
    plt.ylabel("True Clinical Label")
    plt.tight_layout()
    path = out_dir / "figures" / "confusion_matrix.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
