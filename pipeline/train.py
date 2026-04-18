"""
End-to-end training pipeline orchestrator.

Usage:
    # Train all models with default settings
    python -m pipeline.train

    # Train specific models
    python -m pipeline.train --models logistic_regression svm knn

    # Skip grid search (baseline only, much faster)
    python -m pipeline.train --no-grid-search

    # Disable feature engineering
    python -m pipeline.train --no-feature-engineering

    # Custom data path
    python -m pipeline.train --data-path /path/to/data.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.config import (
    MODELS_DIR,
    REPORTS_DIR,
    SCALED_MODELS,
    FeatureEngineeringConfig,
    MODEL_SPECS,
    RAW_FEATURES,
)
from pipeline.data_loader import load_data, split_data, validate_data
from pipeline.evaluation import (
    build_comparison_table,
    evaluate_all,
    save_reports,
)
from pipeline.feature_engineering import FeatureEngineer
from pipeline.models import train_all_models
from pipeline.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_pipeline(
    data_path: str | Path | None = None,
    model_names: list[str] | None = None,
    run_grid_search: bool = True,
    enable_feature_engineering: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Execute the full training pipeline and return the comparison table.

    Steps:
        1. Load & validate data
        2. Train/test split
        3. Feature engineering (optional)
        4. Preprocessing (label encoding + scaling)
        5. Train all models (with optional GridSearchCV)
        6. Evaluate on test set
        7. Save artifacts (models, scaler, encoder, reports)
    """
    setup_logging(verbose)
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("  CROP RECOMMENDATION — END-TO-END TRAINING PIPELINE")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & validate
    # ------------------------------------------------------------------
    logger.info("\n[Step 1/7] Loading and validating data ...")
    df = load_data(path=data_path)
    df = validate_data(df)
    logger.info("  Dataset shape: %s | Classes: %d", df.shape, df["crop"].nunique())

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/7] Splitting data ...")
    X_train, X_test, y_train, y_test = split_data(df)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    if enable_feature_engineering:
        logger.info("\n[Step 3/7] Engineering features ...")
        fe_config = FeatureEngineeringConfig()
        fe = FeatureEngineer(config=fe_config)
        X_train = fe.fit_transform(X_train)
        X_test = fe.transform(X_test)
        logger.info("  Feature count after engineering: %d (was %d)",
                     X_train.shape[1], len(RAW_FEATURES))
    else:
        logger.info("\n[Step 3/7] Feature engineering SKIPPED")
        fe = None

    # ------------------------------------------------------------------
    # 4. Preprocessing (encode + scale)
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/7] Preprocessing ...")
    preprocessor = Preprocessor()
    preprocessor.fit(X_train, y_train)

    y_train_enc = preprocessor.transform_labels(y_train)
    y_test_enc = preprocessor.transform_labels(y_test)

    X_train_scaled = preprocessor.transform_features(X_train)
    X_test_scaled = preprocessor.transform_features(X_test)

    # ------------------------------------------------------------------
    # 5. Override grid search flag per spec if requested
    # ------------------------------------------------------------------
    if not run_grid_search:
        for spec in MODEL_SPECS.values():
            spec.run_grid_search = False

    # ------------------------------------------------------------------
    # 6. Train models
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/7] Training models ...")
    train_results = train_all_models(
        X_train_raw=X_train,
        y_train_encoded=y_train_enc,
        X_train_scaled=X_train_scaled,
        model_names=model_names,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    logger.info("\n[Step 6/7] Evaluating on test set ...")
    eval_results = evaluate_all(
        train_results=train_results,
        X_test_raw=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test_enc,
        scaled_models=SCALED_MODELS,
        label_names=preprocessor.classes,
    )
    comparison = build_comparison_table(eval_results)

    # ------------------------------------------------------------------
    # 8. Save artifacts
    # ------------------------------------------------------------------
    logger.info("\n[Step 7/7] Saving artifacts ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, tr in train_results.items():
        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(tr.estimator, model_path)
        logger.info("  Saved model: %s", model_path)

    joblib.dump(preprocessor.scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(preprocessor.label_encoder, MODELS_DIR / "label_encoder.pkl")
    logger.info("  Saved scaler and label_encoder")

    if fe is not None:
        joblib.dump(fe, MODELS_DIR / "feature_engineer.pkl")
        logger.info("  Saved feature_engineer")

    save_reports(eval_results, comparison, REPORTS_DIR)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE — %.1fs total", elapsed)
    logger.info("=" * 70)
    logger.info("\nModel Comparison (sorted by accuracy):\n")
    logger.info("\n%s", comparison.to_string(index=False))

    best = comparison.iloc[0]
    logger.info(
        "\nBest model: %s (accuracy=%.4f, f1=%.4f)",
        best["model"], best["accuracy"], best["f1"],
    )

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop Recommendation ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to the CSV data file (default: data/train_set_label.csv)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(MODEL_SPECS.keys()),
        help="Train only these models (default: all)",
    )
    parser.add_argument(
        "--no-grid-search", action="store_true",
        help="Skip GridSearchCV; train baselines only (faster)",
    )
    parser.add_argument(
        "--no-feature-engineering", action="store_true",
        help="Disable feature engineering; use raw features only",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        model_names=args.models,
        run_grid_search=not args.no_grid_search,
        enable_feature_engineering=not args.no_feature_engineering,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
