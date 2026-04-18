"""
Data loading, validation, and train/test splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import (
    DATA_URL,
    LOCAL_DATA_PATH,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def load_data(path: Path | None = None, url: str | None = None) -> pd.DataFrame:
    """Load the crop recommendation dataset from a local file or URL.

    Tries the local path first, falls back to the URL, then raises.
    """
    path = Path(path) if path else LOCAL_DATA_PATH
    url = url or DATA_URL

    if path.exists():
        logger.info("Loading data from local file: %s", path)
        df = pd.read_csv(path)
    else:
        logger.info("Local file not found. Downloading from %s", url)
        df = pd.read_csv(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved downloaded data to %s", path)

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run sanity checks and return a cleaned copy."""
    required_cols = set(RAW_FEATURES + [TARGET_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_rows = len(df)
    df = df.dropna(subset=RAW_FEATURES + [TARGET_COL])
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing values", dropped)

    df = df.drop_duplicates()
    deduped = initial_rows - dropped - len(df)
    if deduped:
        logger.warning("Dropped %d duplicate rows", deduped)

    for col in RAW_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Feature column '{col}' is not numeric (dtype={df[col].dtype})")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Target column has fewer than 2 unique classes")

    logger.info(
        "Validated data: %d rows, %d features, %d classes",
        len(df), len(RAW_FEATURES), df[TARGET_COL].nunique(),
    )
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X = df[RAW_FEATURES].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Split: train=%d, test=%d (test_size=%.2f)",
        len(X_train), len(X_test), test_size,
    )
    return X_train, X_test, y_train, y_test
