"""
Crop Recommendation System - Utility Functions
===============================================
Reusable functions for data loading, preprocessing, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(url: str = config.DATA_URL) -> pd.DataFrame:
    """
    Load the crop recommendation dataset.
    
    Parameters
    ----------
    url : str
        URL to the dataset CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(url)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Preprocess the dataset: encode labels, split, and scale features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple containing:
        - X_train_scaled: Scaled training features
        - X_test_scaled: Scaled testing features
        - y_train: Training labels (encoded)
        - y_test: Testing labels (encoded)
        - scaler: Fitted StandardScaler
        - label_encoder: Fitted LabelEncoder
    """
    # Separate features and target
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data preprocessing complete!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder


def get_preprocessed_data() -> Tuple:
    """
    Convenience function to load and preprocess data in one step.
    
    Returns
    -------
    Tuple containing preprocessed data and preprocessing objects
    """
    df = load_data()
    return preprocess_data(df), df


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    cv_folds: int = config.CV_FOLDS
) -> Dict[str, Any]:
    """
    Train a model and compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    model : sklearn estimator
        Machine learning model to train
    X_train, X_test : np.ndarray
        Training and testing features
    y_train, y_test : np.ndarray
        Training and testing labels
    model_name : str
        Name of the model for display
    cv_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    Dict containing:
        - All evaluation metrics
        - Predictions
        - Training time
        - Trained model
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print('='*60)
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"-" * 40)
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n📈 Cross-Validation ({cv_folds}-fold):")
    print(f"   Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n⏱️  Training Time: {training_time:.4f} seconds")
    
    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'training_time': training_time,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str = "Model"
) -> None:
    """Print detailed classification report."""
    print(f"\n{'='*60}")
    print(f"📋 CLASSIFICATION REPORT - {model_name}")
    print('='*60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot and optionally save confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    plt.show()


def plot_roc_curves(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> float:
    """
    Plot ROC curves for multi-class classification.
    
    Returns
    -------
    float
        Macro-average AUC score
    """
    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Compute ROC curve for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(all_fpr, mean_tpr, 'b-', linewidth=2,
             label=f'Macro-average (AUC = {macro_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    plt.show()
    
    return macro_auc


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️ {model_name} doesn't have feature_importances_ attribute")
        return
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names)))
    
    plt.barh([feature_names[i] for i in indices], importance[indices],
             color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    
    for i, (j, v) in enumerate(zip(indices, importance[indices])):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    plt.show()


def plot_cv_scores(
    cv_scores: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot cross-validation scores distribution.
    """
    plt.figure(figsize=figsize)
    
    folds = range(1, len(cv_scores) + 1)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(cv_scores)))
    
    bars = plt.bar(folds, cv_scores, color=colors, edgecolor='black', linewidth=0.5)
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {cv_scores.mean():.4f}')
    plt.fill_between([0.5, len(cv_scores) + 0.5], 
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(),
                     alpha=0.2, color='red', label=f'±1 Std: {cv_scores.std():.4f}')
    
    plt.xlabel('Fold', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    plt.title(f'Cross-Validation Scores - {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(folds)
    plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    plt.show()


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(
    model: Any,
    model_name: str,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    results: Dict[str, Any]
) -> None:
    """
    Save trained model, scaler, encoder, and results.
    """
    model_path = config.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    encoder_path = config.MODELS_DIR / "label_encoder.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    # Save metrics
    metrics = {k: v for k, v in results.items() 
               if k not in ['model', 'y_pred', 'y_pred_proba', 'cv_scores']}
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(config.RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.csv", index=False)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Label encoder saved to: {encoder_path}")


def load_model(model_name: str) -> Tuple[Any, StandardScaler, LabelEncoder]:
    """
    Load trained model, scaler, and label encoder.
    """
    model_path = config.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    encoder_path = config.MODELS_DIR / "label_encoder.pkl"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    
    print(f"✅ Model loaded from: {model_path}")
    return model, scaler, label_encoder


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_crop(
    model: Any,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float
) -> str:
    """
    Predict crop recommendation based on input parameters.
    
    Parameters
    ----------
    model : trained model
    scaler : fitted StandardScaler
    label_encoder : fitted LabelEncoder
    N, P, K : float
        Soil nutrient content (Nitrogen, Phosphorus, Potassium)
    temperature : float
        Temperature in Celsius
    humidity : float
        Relative humidity percentage
    ph : float
        Soil pH value
    rainfall : float
        Rainfall in mm
        
    Returns
    -------
    str
        Recommended crop name
    """
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop = label_encoder.inverse_transform(prediction)[0]
    
    print(f"\n🌾 CROP RECOMMENDATION")
    print("=" * 40)
    print(f"📊 Input Parameters:")
    print(f"   N: {N}, P: {P}, K: {K}")
    print(f"   Temperature: {temperature}°C")
    print(f"   Humidity: {humidity}%")
    print(f"   pH: {ph}")
    print(f"   Rainfall: {rainfall}mm")
    print(f"\n🎯 Recommended Crop: {crop.upper()}")
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(input_scaled)[0]
        top_3_idx = np.argsort(probas)[-3:][::-1]
        print(f"\n📈 Top 3 Recommendations:")
        for i, idx in enumerate(top_3_idx, 1):
            print(f"   {i}. {label_encoder.classes_[idx]}: {probas[idx]*100:.2f}%")
    
    return crop


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def create_results_summary(results: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """
    Create a summary DataFrame of model results.
    """
    summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                   'CV Mean', 'CV Std', 'Training Time (s)'],
        'Value': [
            results['accuracy'],
            results['precision'],
            results['recall'],
            results['f1_score'],
            results['cv_mean'],
            results['cv_std'],
            results['training_time']
        ]
    })
    summary['Model'] = model_name
    return summary
