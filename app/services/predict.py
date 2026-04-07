import hashlib
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
HASH_PATH = PROJECT_ROOT / "artifacts" / "model.sha256"

_model = None
_target_names = None
_metrics = None
_loaded_mtime = None


def load_model() -> None:
    global _model, _target_names, _metrics, _loaded_mtime
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    if _model is not None and _loaded_mtime == model_mtime:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{MODEL_PATH}'. Run ml/train.py first."
        )
    verify_artifact_integrity()

    payload = joblib.load(MODEL_PATH)
    _model = payload["model_pipeline"]
    _target_names = payload["target_names"]
    _metrics = {
        "model_name": payload.get("model_name", "unknown"),
        "accuracy": float(payload.get("accuracy", 0.0)),
        "trained_at": payload.get("trained_at", "unknown"),
        "feature_names": payload.get(
            "feature_names",
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        ),
    }
    _loaded_mtime = model_mtime


def predict_class(features: list[float]) -> tuple[int, str]:
    if _model is None or _target_names is None:
        load_model()

    values = np.array([features], dtype=float)
    class_id = int(_model.predict(values)[0])
    class_name = str(_target_names[class_id])
    return class_id, class_name


def get_model_metrics() -> dict[str, str | float | list[str]]:
    if _metrics is None:
        load_model()
    return _metrics


def verify_artifact_integrity() -> None:
    if not HASH_PATH.exists():
        raise FileNotFoundError(f"Artifact hash file not found at '{HASH_PATH}'.")
    expected_hash = HASH_PATH.read_text(encoding="utf-8").strip()
    actual_hash = calculate_file_sha256(MODEL_PATH)
    if expected_hash != actual_hash:
        raise ValueError("Model artifact integrity check failed.")


def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
