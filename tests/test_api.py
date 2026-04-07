from fastapi.testclient import TestClient

from app.main import app
from ml.train import ARTIFACT_PATH, save_model_artifact, train_knn_model


def _ensure_model_artifacts() -> None:
    if ARTIFACT_PATH.exists() and ARTIFACT_PATH.with_suffix(".sha256").exists():
        return
    training_result = train_knn_model()
    save_model_artifact(training_result, ARTIFACT_PATH)


def test_health_endpoint() -> None:
    _ensure_model_artifacts()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint() -> None:
    _ensure_model_artifacts()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "knn"
    assert isinstance(payload["accuracy"], float)
    assert "trained_at" in payload
    assert len(payload["feature_names"]) == 4


def test_predict_endpoint() -> None:
    _ensure_model_artifacts()
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class_id"] in [0, 1, 2]
    assert isinstance(payload["predicted_class"], str)
