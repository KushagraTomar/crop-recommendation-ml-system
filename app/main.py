from fastapi import FastAPI

from app.api.routes.predict import router as prediction_router
from app.api.routes.web import router as web_router
from app.core.settings import settings

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.include_router(web_router)
app.include_router(prediction_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
