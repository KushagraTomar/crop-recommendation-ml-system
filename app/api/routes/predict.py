from fastapi import APIRouter

from app.schemas.prediction import CropInput, PredictionResponse
from app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/api/v1", tags=["prediction"])
registry = ModelRegistry()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: CropInput) -> PredictionResponse:
    predictions, unavailable = registry.predict_all(payload)
    return PredictionResponse(predictions=predictions, unavailable_models=unavailable)


@router.get("/models")
def model_status() -> dict:
    registry.refresh()
    return {"models": registry.model_status}
