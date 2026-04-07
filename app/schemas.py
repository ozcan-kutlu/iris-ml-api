from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int


class MetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    trained_at: str
    feature_names: list[str]
