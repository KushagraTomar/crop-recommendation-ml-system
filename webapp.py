"""
FastAPI web app for crop recommendation across multiple models.
"""

from typing import Dict, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from inference_service import CropInput, InferenceService

app = FastAPI(title="Crop Recommendation - Multi Model Inference")
templates = Jinja2Templates(directory="templates")
service = InferenceService()


class CropInputRequest(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)


@app.on_event("startup")
def startup_event() -> None:
    service.ensure_models_ready()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predictions": None,
            "input_data": None,
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
) -> HTMLResponse:
    input_data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    predictions: Optional[Dict[str, str]] = None
    error: Optional[str] = None

    try:
        payload = CropInput(**input_data)
        predictions = service.predict_all(payload)
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predictions": predictions,
            "input_data": input_data,
            "error": error,
        },
    )


@app.post("/api/predict")
def predict_api(payload: CropInputRequest) -> Dict[str, Dict[str, str]]:
    predictions = service.predict_all(CropInput(**payload.model_dump()))
    return {"predictions": predictions}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
