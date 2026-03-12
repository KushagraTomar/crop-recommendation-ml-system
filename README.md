# Crop Recommendation ML System

A machine learning project that recommends suitable crops based on soil nutrients and environmental conditions.  
The repository includes model training notebooks, serialized model artifacts, and a FastAPI application (REST API + web interface) for inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Run the Application](#run-the-application)
- [API Endpoints](#api-endpoints)
- [Input Schema](#input-schema)
- [Model Artifacts](#model-artifacts)
- [Notebooks and Training Workflow](#notebooks-and-training-workflow)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

This project predicts crop recommendations using multiple ML models and serves results through:

- A web form for interactive predictions
- A JSON API for integration with other systems
- A model registry service that loads all available `.pkl` models and returns predictions from each model

The goal is to compare model outputs and provide reliable inference from a consistent input schema.

## Features

- Multi-model prediction (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting)
- FastAPI-based REST API
- Jinja2-powered web interface
- Shared preprocessing support (scaler + label encoder)
- Model availability reporting (`loaded`, missing file, load failure, prediction failure)
- Health check endpoint for service monitoring

## Tech Stack

- Python
- scikit-learn
- XGBoost
- FastAPI + Uvicorn
- Jinja2 templates
- Pandas / NumPy
- Jupyter Notebook

## Project Structure

```text
crop-recommendation-ml-system/
├── app/
│   ├── api/routes/          # REST and web routes
│   ├── core/settings.py     # App/model path configuration
│   ├── schemas/             # Pydantic input/output schemas
│   ├── services/            # Model registry and prediction logic
│   ├── templates/           # Web UI template
│   └── main.py              # FastAPI app entrypoint
├── models/                  # Serialized model artifacts (.pkl)
├── notebooks/
│   ├── eda/                 # Exploratory analysis notebook(s)
│   └── models/              # Model training notebooks
├── requirements.txt
└── README.md
```

## How It Works

1. Request input includes 7 features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.
2. `ModelRegistry` loads:
   - all model files defined in `app/core/settings.py`
   - optional `scaler.pkl`
   - optional `label_encoder.pkl`
3. Inputs are prepared in both raw and scaled form.
4. Scaled models (`logistic_regression`, `knn`, `svm`, `naive_bayes`) use scaled features; other models use raw features.
5. Predictions are returned as human-readable crop labels when `label_encoder.pkl` is available.

## Getting Started

### 1) Clone the repository

```bash
git clone <your-repository-url>
cd crop-recommendation-ml-system
```

### 2) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the FastAPI server from the project root:

```bash
uvicorn app.main:app --reload
```

Then open:

- Web UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## API Endpoints

### `POST /api/v1/predict`

Returns predictions from all currently available models.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.9,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
  }'
```

Example response:

```json
{
  "predictions": {
    "logistic_regression": "rice",
    "random_forest": "rice"
  },
  "unavailable_models": {
    "svm": "Missing file: svm.pkl"
  }
}
```

### `GET /api/v1/models`

Returns load status of all configured model files.

### `GET /health`

Simple health endpoint:

```json
{"status": "ok"}
```

## Input Schema

| Field | Type | Constraints |
|---|---|---|
| `N` | float | `>= 0` |
| `P` | float | `>= 0` |
| `K` | float | `>= 0` |
| `temperature` | float | no strict range |
| `humidity` | float | `0` to `100` |
| `ph` | float | `0` to `14` |
| `rainfall` | float | `>= 0` |

## Model Artifacts

The app expects serialized artifacts inside `models/`:

- `logistic_regression.pkl`
- `decision_tree.pkl`
- `random_forest.pkl`
- `svm.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `gradient_boosting.pkl`
- `scaler.pkl` (optional but recommended)
- `label_encoder.pkl` (optional but recommended)

If any model file is missing or invalid, the service continues running and reports the issue in `unavailable_models`.

## Notebooks and Training Workflow

The `notebooks/models/` directory contains separate notebooks for each model:

- `01_logistic_regression.ipynb`
- `02_decision_tree.ipynb`
- `03_random_forest.ipynb`
- `04_svm.ipynb`
- `05_knn.ipynb`
- `06_naive_bayes.ipynb`
- `07_gradient_boosting.ipynb`

Typical workflow:

1. Explore data in `notebooks/eda/01_eda.ipynb`
2. Train/tune model in a model notebook
3. Export model to `models/*.pkl`
4. Export shared preprocessing artifacts (`scaler.pkl`, `label_encoder.pkl`) if needed
5. Run the API and verify predictions

## Dataset

Notebooks reference this dataset URL:

- [https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv](https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv)

You can also adapt notebooks to use a local file in `data/` if you prefer offline workflows.

## Troubleshooting

- **`unavailable_models` is not empty**  
  Ensure all required `.pkl` files exist in `models/` and were saved correctly.

- **Validation errors on input**  
  Check field constraints for `humidity`, `ph`, and non-negative nutrient/rainfall values.

- **Import/module errors**  
  Confirm you are running commands from project root and the virtual environment is activated.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
