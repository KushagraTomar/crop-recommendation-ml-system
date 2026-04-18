"""
Central configuration for the training pipeline.

All hyperparameter grids, feature definitions, model registry,
and path settings live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "crop"

DATA_URL = (
    "https://raw.githubusercontent.com/dphi-official/Datasets/"
    "master/crop_recommendation/train_set_label.csv"
)
LOCAL_DATA_PATH = DATA_DIR / "train_set_label.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

SCALED_MODELS = {"logistic_regression", "svm", "knn", "naive_bayes"}


@dataclass
class ModelSpec:
    """Defines a model, its baseline parameters, and tuning grid."""

    name: str
    estimator_class: type
    baseline_params: Dict[str, Any]
    param_grid: Dict[str, List[Any]]
    needs_scaling: bool = False
    run_grid_search: bool = True


MODEL_SPECS: Dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        estimator_class=LogisticRegression,
        baseline_params=dict(max_iter=1000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE),
        param_grid=dict(
            C=[0.01, 0.1, 1.0, 10.0],
            solver=["lbfgs", "liblinear"],
            max_iter=[1000],
        ),
        needs_scaling=True,
    ),
    "decision_tree": ModelSpec(
        name="decision_tree",
        estimator_class=DecisionTreeClassifier,
        baseline_params=dict(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            criterion="gini", random_state=RANDOM_STATE,
        ),
        param_grid=dict(
            max_depth=[5, 10, 20, 50],
            min_samples_split=[2, 5, 7],
            min_samples_leaf=[2, 4],
            criterion=["gini", "entropy"],
        ),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        estimator_class=RandomForestClassifier,
        baseline_params=dict(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", bootstrap=True,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        param_grid=dict(
            n_estimators=[10, 25, 50, 75, 100],
            max_depth=[5, 10, 15, 20],
            min_samples_split=[2, 5, 7],
            min_samples_leaf=[2, 4],
            max_features=["sqrt", "log2", None],
        ),
    ),
    "svm": ModelSpec(
        name="svm",
        estimator_class=SVC,
        baseline_params=dict(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        param_grid=dict(
            C=[0.1, 1, 10, 100],
            gamma=["scale", "auto", 0.01, 0.1, 1],
            kernel=["poly", "rbf", "linear", "sigmoid"],
        ),
        needs_scaling=True,
    ),
    "knn": ModelSpec(
        name="knn",
        estimator_class=KNeighborsClassifier,
        baseline_params=dict(n_neighbors=5, weights="uniform", algorithm="auto", metric="euclidean"),
        param_grid=dict(
            n_neighbors=[3, 5, 7, 9, 11, 15, 21, 25],
            weights=["uniform", "distance"],
            algorithm=["auto", "ball_tree", "kd_tree", "brute"],
            metric=["euclidean", "manhattan", "minkowski"],
        ),
    ),
    "naive_bayes": ModelSpec(
        name="naive_bayes",
        estimator_class=GaussianNB,
        baseline_params=dict(var_smoothing=1e-9),
        param_grid=dict(
            var_smoothing=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        ),
        needs_scaling=True,
    ),
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        estimator_class=GradientBoostingClassifier,
        baseline_params=dict(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE,
        ),
        param_grid=dict(
            n_estimators=[10, 20, 50, 100],
            learning_rate=[0.05, 0.1, 0.2],
            max_depth=[3, 5, 7],
            min_samples_split=[2, 5],
            min_samples_leaf=[2, 4],
        ),
    ),
}


@dataclass
class FeatureEngineeringConfig:
    """Toggle-able feature engineering steps."""

    add_npk_total: bool = True
    add_npk_ratios: bool = True
    add_temp_humidity_interaction: bool = True
    add_rainfall_bins: bool = True
    add_ph_category: bool = True
    add_polynomial_features: bool = False
    polynomial_degree: int = 2
    polynomial_columns: List[str] = field(
        default_factory=lambda: ["temperature", "humidity", "ph", "rainfall"]
    )
