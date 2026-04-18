"""
Model evaluation and comparison reporting.

Generates per-model metrics, a comparison table, and persists
reports as CSV / JSON for later review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from pipeline.models import TrainResult

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: List[List[int]]
    classification_report: str


def evaluate_model(
    train_result: TrainResult,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_names: np.ndarray | None = None,
) -> EvalResult:
    """Evaluate a trained model on the hold-out test set."""
    y_pred = train_result.estimator.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0,
    )

    logger.info(
        "  %s — acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
        train_result.model_name, acc, prec, rec, f1,
    )

    return EvalResult(
        model_name=train_result.model_name,
        accuracy=acc,
        precision_weighted=prec,
        recall_weighted=rec,
        f1_weighted=f1,
        confusion_matrix=cm.tolist(),
        classification_report=report,
    )


def evaluate_all(
    train_results: Dict[str, TrainResult],
    X_test_raw: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_test: np.ndarray,
    scaled_models: set[str],
    label_names: np.ndarray | None = None,
) -> Dict[str, EvalResult]:
    """Evaluate every trained model, routing to raw or scaled test features."""
    logger.info("Evaluating %d models on test set ...", len(train_results))
    evals: Dict[str, EvalResult] = {}
    for name, tr in train_results.items():
        X = X_test_scaled if name in scaled_models else X_test_raw
        evals[name] = evaluate_model(tr, X, y_test, label_names)
    return evals


def build_comparison_table(evals: Dict[str, EvalResult]) -> pd.DataFrame:
    """Build a sorted comparison DataFrame across all models."""
    rows = []
    for ev in evals.values():
        rows.append({
            "model": ev.model_name,
            "accuracy": round(ev.accuracy, 4),
            "precision": round(ev.precision_weighted, 4),
            "recall": round(ev.recall_weighted, 4),
            "f1": round(ev.f1_weighted, 4),
        })
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df


def save_reports(
    evals: Dict[str, EvalResult],
    comparison_df: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Persist evaluation reports to disk."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved comparison table to %s", comparison_path)

    for ev in evals.values():
        model_dir = reports_dir / ev.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        report_path = model_dir / "classification_report.txt"
        report_path.write_text(ev.classification_report)

        metrics_path = model_dir / "metrics.json"
        metrics = {
            "accuracy": ev.accuracy,
            "precision_weighted": ev.precision_weighted,
            "recall_weighted": ev.recall_weighted,
            "f1_weighted": ev.f1_weighted,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

        cm_path = model_dir / "confusion_matrix.json"
        cm_path.write_text(json.dumps(ev.confusion_matrix))

    logger.info("Saved per-model reports to %s", reports_dir)
