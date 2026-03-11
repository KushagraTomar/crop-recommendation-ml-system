"""
Crop Recommendation System - Configuration File
================================================
Central configuration for all project parameters.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_URL = "https://raw.githubusercontent.com/Gladiator07/Crop-Recommendation-System/main/Data/Crop_recommendation.csv"
KAGGLE_DATASET = "atharvaingle/crop-recommendation-dataset"

# Feature columns
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COLUMN = 'label'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model hyperparameters
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'multi_class': 'multinomial',
        'solver': 'lbfgs',
        'C': 1.0
    },
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss'
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        'n_jobs': -1
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'probability': True
    },
    'naive_bayes': {}
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'
COLOR_PALETTE = 'husl'

# Plot sizes
FIGSIZE_SMALL = (8, 6)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_LARGE = (16, 10)
FIGSIZE_WIDE = (18, 6)
