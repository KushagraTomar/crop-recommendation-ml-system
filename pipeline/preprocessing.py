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


logger = logging.getLogger(__name__)


class Preprocessor:
    """Handles label encoding and feature scaling with fit/transform semantics."""

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """Fit encoder on labels and scaler on all training features."""
        self.label_encoder.fit(y_train)
        self.feature_columns = list(X_train.columns)
        self.scaler.fit(X_train)
        self.is_fitted = True

        logger.info(
            "Preprocessor fitted: %d classes, %d features scaled",
            len(self.label_encoder.classes_), len(self.feature_columns),
        )
        return self

    def transform_labels(self, y: pd.Series):
        return self.label_encoder.transform(y)

    def inverse_transform_labels(self, y_encoded: np.ndarray):
        return self.label_encoder.inverse_transform(y_encoded)

    def transform_features(self, X: pd.DataFrame):
        """Scale all features using the columns seen during fit."""
        scaled = self.scaler.transform(X[self.feature_columns])
        return pd.DataFrame(scaled, columns=self.feature_columns, index=X.index)

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """Convenience: fit then transform training data."""
        self.fit(X_train, y_train)
        X_scaled = self.transform_features(X_train)
        y_encoded = self.transform_labels(y_train)
        return X_scaled, y_encoded
