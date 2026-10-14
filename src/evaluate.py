"""Evaluation on the held-out test split: metrics, confusion matrix, ROC, error analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config
from .data import labels_from_dataset, load_test_dataset
from .predict import load_trained_model, model_img_size
from .utils import ensure_dir, save_json
from .visualize import plot_confusion_matrix, plot_prediction_grid, plot_roc


def compute_metrics(y_true, probs, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Pure function (no TensorFlow): metrics for binary labels and probabilities."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if len(y_true) != len(probs):
        raise ValueError(f"y_true ({len(y_true)}) and probs ({len(probs)}) differ in length")
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if both_classes else None,
        "confusion_matrix": cm.tolist(),
        "tn_fp_fn_tp": [tn, fp, fn, tp],
    }


def results_markdown(m: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.2f}%"

    tn, fp, fn, tp = m["tn_fp_fn_tp"]
    auc_text = "n/a" if m["roc_auc"] is None else format(m["roc_auc"], ".4f")
    return "\n".join(
        [
            "| Metric | Value |",
            "|---|---|",
            f"| Test images | {m['n_samples']} |",
            f"| Accuracy | {pct(m['accuracy'])} |",
            f"| Sensitivity (recall for PNEUMONIA) | {pct(m['recall_sensitivity'])} |",
            f"| Specificity (recall for NORMAL) | {pct(m['specificity'])} |",
            f"| Precision | {pct(m['precision'])} |",
            f"| F1 score | {pct(m['f1'])} |",
            f"| ROC-AUC | {auc_text} |",
            f"| True neg / False pos / False neg / True pos | {tn} / {fp} / {fn} / {tp} |",
            f"| Decision threshold | {m['threshold']} |",
        ]
    )


def evaluate(
    data_dir=None,
    model_path=config.MODEL_PATH,
    outputs_dir=config.OUTPUTS_DIR,
    batch_size: int = config.BATCH_SIZE,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    outputs_dir = ensure_dir(outputs_dir)
    model = load_trained_model(model_path)
    test_ds, files = load_test_dataset(data_dir, model_img_size(model), batch_size)

    y_true = labels_from_dataset(test_ds)
    probs = model.predict(test_ds, verbose=0).reshape(-1)
    assert len(y_true) == len(probs) == len(files), "prediction/label/file count mismatch"

    metrics = compute_metrics(y_true, probs, threshold)
    metrics["threshold_sweep"] = {
        str(t): {k: compute_metrics(y_true, probs, t)[k]
                 for k in ("accuracy", "recall_sensitivity", "specificity", "precision")}
        for t in (0.3, 0.4, 0.5, 0.6, 0.7)
    }

    y_pred = (probs >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=config.CLASS_NAMES, zero_division=0
    )
    print("\n" + report)
    print(f"Confusion matrix [[TN FP] [FN TP]]: {metrics['confusion_matrix']}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%   ROC-AUC: {metrics['roc_auc']}")

    # ---- artefacts
    save_json(metrics, outputs_dir / "metrics.json")
    (outputs_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (outputs_dir / "results_table.md").write_text(results_markdown(metrics) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "file": [str(f) for f in files],
            "actual": [config.CLASS_NAMES[i] for i in y_true],
            "probability_pneumonia": probs.round(5),
            "predicted": [config.CLASS_NAMES[i] for i in y_pred],
            "correct": y_pred == y_true,
        }
    ).to_csv(outputs_dir / "test_predictions.csv", index=False)

    plot_confusion_matrix(metrics["confusion_matrix"], outputs_dir / "confusion_matrix.png")
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        plot_roc(fpr, tpr, metrics["roc_auc"], outputs_dir / "roc_curve.png")
    plot_prediction_grid(files, y_true, probs, outputs_dir / "predictions_sample.png",
                         threshold, title="Predicted vs actual (random test images)")
    plot_prediction_grid(files, y_true, probs, outputs_dir / "misclassified.png", threshold,
                         only_wrong=True, title="Misclassified test images (error analysis)")
    print(f"Saved metrics + figures to {outputs_dir}")
    return metrics
