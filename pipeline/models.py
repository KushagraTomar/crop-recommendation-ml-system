"""
Model training and hyperparameter tuning.

Provides a unified interface for training any model defined in config.MODEL_SPECS,
with optional GridSearchCV tuning and cross-validation scoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from pipeline.config import CV_FOLDS, MODEL_SPECS, RANDOM_STATE, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Captures everything about a single model's training run."""

    model_name: str
    estimator: Any
    best_params: Dict[str, Any]
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    grid_search_results: Optional[Dict[str, Any]] = None
    train_duration_sec: float = 0.0


def train_single_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    n_jobs: int = -1,
) -> TrainResult:
    """Train one model: baseline CV, optional GridSearchCV, return best."""
    logger.info("=" * 60)
    logger.info("Training: %s", spec.name)
    logger.info("=" * 60)

    t0 = time.time()

    baseline = spec.estimator_class(**spec.baseline_params)
    cv_scores = cross_val_score(
        baseline, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=n_jobs,
    )
    logger.info(
        "  Baseline CV accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std(),
    )

    best_estimator = baseline
    best_params = spec.baseline_params.copy()
    grid_results = None

    if spec.run_grid_search and spec.param_grid:

        base_params = {"random_state": RANDOM_STATE} if "random_state" in spec.baseline_params else {}
        if spec.name == "svm":
            base_params["probability"] = True

        gs = GridSearchCV(
            spec.estimator_class(**base_params),
            param_grid=spec.param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=0,
            refit=True,
        )
        gs.fit(X_train, y_train)

        best_estimator = gs.best_estimator_
        best_params = gs.best_params_
        grid_results = {
            "best_score": gs.best_score_,
            "best_params": gs.best_params_,
            "cv_results": gs.cv_results_,
        }
        logger.info("  Best grid score: %.4f | params: %s", gs.best_score_, gs.best_params_)
    else:
        best_estimator.fit(X_train, y_train)

    duration = time.time() - t0
    logger.info("  Completed in %.1fs", duration)

    return TrainResult(
        model_name=spec.name,
        estimator=best_estimator,
        best_params=best_params,
        cv_scores=cv_scores,
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        grid_search_results=grid_results,
        train_duration_sec=duration,
    )


def train_all_models(
    X_train_raw: pd.DataFrame,
    y_train_encoded: np.ndarray,
    X_train_scaled: pd.DataFrame,
    model_names: List[str] | None = None,
    cv_folds: int = CV_FOLDS,
    n_jobs: int = -1,
) -> Dict[str, TrainResult]:
    """Train all (or selected) models, routing each to raw or scaled features."""
    specs = MODEL_SPECS
    if model_names:
        specs = {k: v for k, v in specs.items() if k in model_names}

    results: Dict[str, TrainResult] = {}
    for name, spec in specs.items():
        X = X_train_scaled if spec.needs_scaling else X_train_raw
        result = train_single_model(spec, X, y_train_encoded, cv_folds, n_jobs)
        results[name] = result

    return results