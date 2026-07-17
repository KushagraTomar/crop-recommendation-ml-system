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


def load_data():
    """Load the crop recommendation dataset from a local file or URL.

    Tries the local path first, falls back to the URL, then raises.
    """
    path = LOCAL_DATA_PATH
    url = DATA_URL

    if path.exists():
        logger.info("[info] Loading data from local file: %s", path)
        df = pd.read_csv(path)
    else:
        logger.info("[info] Local file not found. Downloading from %s", url)
        df = pd.read_csv(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("[info] Saved downloaded data to %s", path)

    return df


def validate_data(df: pd.DataFrame):
    """Run sanity checks and return a cleaned copy."""
    required_cols = set(RAW_FEATURES + [TARGET_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_rows = len(df)
    df = df.dropna(subset=RAW_FEATURES + [TARGET_COL])
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning("[warning] Dropped %d rows with missing values", dropped)

    df = df.drop_duplicates()
    deduped = initial_rows - dropped - len(df)
    if deduped:
        logger.warning("[warning] Dropped %d duplicate rows", deduped)

    logger.info(
        "[info] Validated data: %d rows, %d features, %d classes",
        len(df), len(RAW_FEATURES), df[TARGET_COL].nunique(),
    )
    #drop old index and reset index
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
        "[info] Split: train=%d, test=%d (test_size=%.2f)",
        len(X_train), len(X_test), test_size,
    )
    return X_train, X_test, y_train, y_test
