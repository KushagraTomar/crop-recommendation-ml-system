from pathlib import Path


class Settings:
    APP_NAME = "Crop Recommendation API"
    APP_VERSION = "1.0.0"

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"

    MODEL_FILES = {
        "logistic_regression": "logistic_regression.pkl",
        "decision_tree": "decision_tree.pkl",
        "random_forest": "random_forest.pkl",
        "svm": "svm.pkl",
        "knn": "knn.pkl",
        "naive_bayes": "naive_bayes.pkl",
        "gradient_boosting": "gradient_boosting.pkl",
    }

    SCALER_FILE = "scaler.pkl"
    LABEL_ENCODER_FILE = "label_encoder.pkl"


settings = Settings()
