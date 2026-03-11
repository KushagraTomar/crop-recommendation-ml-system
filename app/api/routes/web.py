from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.settings import settings
from app.schemas.prediction import CropInput
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
registry = ModelRegistry()


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predictions": None,
            "unavailable_models": None,
            "input_data": None,
            "error": None,
        },
    )


@router.post("/predict", response_class=HTMLResponse)
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
    error = None
    predictions = None
    unavailable_models = None

    try:
        payload = CropInput(**input_data)
        predictions, unavailable_models = registry.predict_all(payload)
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predictions": predictions,
            "unavailable_models": unavailable_models,
            "input_data": input_data,
            "error": error,
        },
    )
