from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

from app.core.settings import settings
from app.schemas.prediction import CropInput


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
                self.models[model_name] = joblib.load(model_path)
                self.model_status[model_name] = "loaded"
            except Exception as exc:
                self.model_status[model_name] = f"Failed to load: {exc}"

    def _prepare_inputs(self, payload: CropInput) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [[payload.N, payload.P, payload.K, payload.temperature, payload.humidity, payload.ph, payload.rainfall]]
        )
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        return X, X_scaled

    def predict_all(self, payload: CropInput) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Return predictions for loaded models plus unavailable-model reasons.
        """
        self.refresh()
        X_raw, X_scaled = self._prepare_inputs(payload)

        predictions: Dict[str, str] = {}
        unavailable = {k: v for k, v in self.model_status.items() if v != "loaded"}

        # Models that were trained on scaled features in notebooks.
        scaled_models = {"logistic_regression", "knn", "svm", "naive_bayes"}

        for model_name, model in self.models.items():
            model_input = X_scaled if model_name in scaled_models else X_raw
            pred_value = model.predict(model_input)[0]

            if self.label_encoder is not None:
                pred_label = self.label_encoder.inverse_transform([pred_value])[0]
            else:
                pred_label = pred_value

            predictions[model_name] = str(pred_label)

        return predictions, unavailable
