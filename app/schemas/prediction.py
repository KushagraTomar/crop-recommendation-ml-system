from typing import Dict

from pydantic import BaseModel, Field


class CropInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predictions: Dict[str, str]
    unavailable_models: Dict[str, str]
