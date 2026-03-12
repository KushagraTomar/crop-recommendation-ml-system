from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.settings import settings
from app.schemas.prediction import CropInput

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class ModelRegistry:
    """Loads persisted artifacts and serves multi-model predictions."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models: Dict[str, object] = {}
        self.model_status: Dict[str, str] = {}
        self.scaler = None
        self.label_encoder = None

    def refresh(self) -> None:
        """Load available models and shared preprocessing artifacts."""
        self.models = {}
        self.model_status = {}

        scaler_path = self.models_dir / settings.SCALER_FILE
        encoder_path = self.models_dir / settings.LABEL_ENCODER_FILE

        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None

        for model_name, filename in settings.MODEL_FILES.items():
            model_path = self.models_dir / filename
            if not model_path.exists():
                self.model_status[model_name] = f"Missing file: {filename}"
                continue

            try:
                loaded_obj = joblib.load(model_path)
                if loaded_obj is None:
                    self.model_status[model_name] = "Loaded artifact is None"
                    continue
                if not hasattr(loaded_obj, "predict"):
                    self.model_status[model_name] = "Loaded artifact does not implement predict()"
                    continue
                self.models[model_name] = loaded_obj
                self.model_status[model_name] = "loaded"
            except Exception as exc:
                self.model_status[model_name] = f"Failed to load: {exc}"

    def _prepare_inputs(self, payload: CropInput) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_raw_df = pd.DataFrame(
            [
                {
                    "N": payload.N,
                    "P": payload.P,
                    "K": payload.K,
                    "temperature": payload.temperature,
                    "humidity": payload.humidity,
                    "ph": payload.ph,
                    "rainfall": payload.rainfall,
                }
            ],
            columns=FEATURE_COLUMNS,
        )
        print("X_raw_df", X_raw_df)
        if self.scaler is not None:
            scaler_input = X_raw_df if hasattr(self.scaler, "feature_names_in_") else X_raw_df.to_numpy()
            X_scaled = self.scaler.transform(scaler_input)
            X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
        else:
            X_scaled_df = X_raw_df.copy()
            
        print("X_scaled_df", X_scaled_df)
        return X_raw_df, X_scaled_df

    @staticmethod
    def _input_for_model(model: object, X_df: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)
            return X_df[expected_columns]
        return X_df.to_numpy()

    def predict_all(self, payload: CropInput) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Return predictions for loaded models plus unavailable-model reasons.
        """
        self.refresh()
        X_raw_df, X_scaled_df = self._prepare_inputs(payload)

        predictions: Dict[str, str] = {}
        unavailable = {k: v for k, v in self.model_status.items() if v != "loaded"}

        # Models that were trained on scaled features in notebooks.
        scaled_models = {"logistic_regression", "knn", "svm", "naive_bayes"}

        for model_name, model in self.models.items():
            try:
                candidate_df = X_scaled_df if model_name in scaled_models else X_raw_df
                model_input = self._input_for_model(model, candidate_df)
                pred_value = model.predict(model_input)[0]

                if self.label_encoder is not None:
                    pred_label = self.label_encoder.inverse_transform([pred_value])[0]
                else:
                    pred_label = pred_value

                predictions[model_name] = str(pred_label)
            except Exception as exc:
                unavailable[model_name] = f"Prediction failed: {exc}"

        return predictions, unavailable
