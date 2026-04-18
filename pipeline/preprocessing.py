"""
Data preprocessing: label encoding and feature scaling.

Fits on training data and transforms both train and test consistently.
Artifacts (scaler, label_encoder) are saved for inference reuse.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pipeline.config import RAW_FEATURES

logger = logging.getLogger(__name__)


class Preprocessor:
    """Handles label encoding and feature scaling with fit/transform semantics."""

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "Preprocessor":
        """Fit encoder on labels and scaler on training features."""
        self.label_encoder.fit(y_train)
        self.scaler.fit(X_train[RAW_FEATURES])
        self._is_fitted = True

        logger.info(
            "Preprocessor fitted: %d classes, %d features scaled",
            len(self.label_encoder.classes_), len(RAW_FEATURES),
        )
        return self

    def transform_labels(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features; returns DataFrame preserving column names."""
        scaled = self.scaler.transform(X[RAW_FEATURES])
        return pd.DataFrame(scaled, columns=RAW_FEATURES, index=X.index)

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Convenience: fit then transform training data."""
        self.fit(X_train, y_train)
        X_scaled = self.transform_features(X_train)
        y_encoded = self.transform_labels(y_train)
        return X_scaled, y_encoded

    @property
    def classes(self) -> np.ndarray:
        return self.label_encoder.classes_
