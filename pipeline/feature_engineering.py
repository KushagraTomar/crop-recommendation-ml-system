"""
Feature engineering transforms applied after loading and before model training.

Each transform is toggle-able via FeatureEngineeringConfig. All transforms
operate on DataFrames and return augmented copies so the original is untouched.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from pipeline.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates derived features from the raw soil/climate columns."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self._poly: PolynomialFeatures | None = None
        self._poly_feature_names: List[str] | None = None
        self._rainfall_bin_edges: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Learn any stateful transforms (polynomial features, bin edges)."""
        if self.config.add_polynomial_features:
            cols = [c for c in self.config.polynomial_columns if c in X.columns]
            self._poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                include_bias=False,
                interaction_only=False,
            )
            self._poly.fit(X[cols])
            self._poly_feature_names = [
                f"poly_{name}"
                for name in self._poly.get_feature_names_out(cols)
            ]

        if self.config.add_rainfall_bins:
            _, self._rainfall_bin_edges = pd.cut(
                X["rainfall"], bins=5, retbins=True, labels=False,
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature transforms."""
        df = X.copy()

        if self.config.add_npk_total:
            df["npk_total"] = df["N"] + df["P"] + df["K"]

        if self.config.add_npk_ratios:
            eps = 1e-8
            df["n_to_p"] = df["N"] / (df["P"] + eps)
            df["n_to_k"] = df["N"] / (df["K"] + eps)
            df["p_to_k"] = df["P"] / (df["K"] + eps)

        if self.config.add_temp_humidity_interaction:
            df["temp_x_humidity"] = df["temperature"] * df["humidity"]

        if self.config.add_rainfall_bins and self._rainfall_bin_edges is not None:
            df["rainfall_bin"] = pd.cut(
                df["rainfall"],
                bins=self._rainfall_bin_edges,
                labels=False,
                include_lowest=True,
            ).fillna(0).astype(int)

        if self.config.add_ph_category:
            df["ph_acidic"] = (df["ph"] < 6.5).astype(int)
            df["ph_neutral"] = ((df["ph"] >= 6.5) & (df["ph"] <= 7.5)).astype(int)
            df["ph_alkaline"] = (df["ph"] > 7.5).astype(int)

        if self.config.add_polynomial_features and self._poly is not None:
            cols = [c for c in self.config.polynomial_columns if c in X.columns]
            poly_arr = self._poly.transform(df[cols])
            poly_df = pd.DataFrame(
                poly_arr, columns=self._poly_feature_names, index=df.index,
            )
            orig_cols = [f"poly_{c}" for c in cols]
            poly_df = poly_df.drop(columns=[c for c in orig_cols if c in poly_df.columns])
            df = pd.concat([df, poly_df], axis=1)

        new_cols = set(df.columns) - set(X.columns)
        if new_cols:
            logger.info("Added %d engineered features: %s", len(new_cols), sorted(new_cols))

        return df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def feature_names(self) -> List[str]:
        """Return names of the last transform output (call after transform)."""
        return list(self._poly_feature_names or [])
