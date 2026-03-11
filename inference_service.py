"""
Inference service for crop recommendation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
import utils


@dataclass
class CropInput:
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class InferenceService:
    """Loads or trains models, then serves predictions."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or config.MODELS_DIR
        self.scaler = None
        self.label_encoder = None
        self.models: Dict[str, object] = {}

    def _model_builders(self) -> Dict[str, object]:
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=config.RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5, n_jobs=-1, random_state=config.RANDOM_STATE
            ),
            "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
            "svm": SVC(kernel="rbf", C=1.0, probability=True, random_state=config.RANDOM_STATE),
            "naive_bayes": GaussianNB(),
            "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        }

    def ensure_models_ready(self) -> None:
        """Load models from disk, or train and save if missing."""
        scaler_path = self.models_dir / "scaler.pkl"
        encoder_path = self.models_dir / "label_encoder.pkl"
        model_paths = {
            model_name: self.models_dir / f"{model_name}.pkl"
            for model_name in self._model_builders().keys()
        }

        all_artifacts_present = (
            scaler_path.exists()
            and encoder_path.exists()
            and all(path.exists() for path in model_paths.values())
        )

        if not all_artifacts_present:
            self._train_and_save_all(model_paths, scaler_path, encoder_path)

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.models = {name: joblib.load(path) for name, path in model_paths.items()}

    def _train_and_save_all(self, model_paths: Dict[str, Path], scaler_path: Path, encoder_path: Path) -> None:
        """Train all baseline models and persist them."""
        df = utils.load_data()
        X_train, _, y_train, _, scaler, label_encoder = utils.preprocess_data(df)

        for model_name, model in self._model_builders().items():
            model.fit(X_train, y_train)
            joblib.dump(model, model_paths[model_name])

        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)

    def predict_all(self, payload: CropInput) -> Dict[str, str]:
        if not self.models or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Models are not loaded. Call ensure_models_ready first.")

        input_values = np.array(
            [[payload.N, payload.P, payload.K, payload.temperature, payload.humidity, payload.ph, payload.rainfall]]
        )
        input_scaled = self.scaler.transform(input_values)

        predictions: Dict[str, str] = {}
        for model_name, model in self.models.items():
            pred_idx = model.predict(input_scaled)[0]
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            predictions[model_name] = str(pred_label)

        return predictions
