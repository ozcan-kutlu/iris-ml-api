from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import MetricsResponse, PredictRequest, PredictResponse
from app.services.predict import get_model_metrics, load_model, predict_class


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except (FileNotFoundError, ValueError):
        # API starts even if model artifact is unavailable.
        pass
    yield


app = FastAPI(title="Iris Classifier API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        class_id, class_name = predict_class(
            [
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        predicted_class=class_name,
        predicted_class_id=class_id,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        data = get_model_metrics()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MetricsResponse(**data)
