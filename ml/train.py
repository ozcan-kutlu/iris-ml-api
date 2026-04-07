import argparse
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.joblib"
LOGGER = logging.getLogger(__name__)


class TrainResult(TypedDict):
    model_name: str
    score: float
    model_pipeline: Pipeline
    target_names: Any
    feature_names: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KNN model on Iris dataset.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=ARTIFACT_PATH,
        help="Output path for joblib artifact.",
    )
    return parser.parse_args()


def train_knn_model(
    random_state: int = 42, test_size: float = 0.2, n_neighbors: int = 5
) -> TrainResult:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")

    iris_dataset = load_iris()
    train_features, test_features, train_labels, test_labels = train_test_split(
        iris_dataset.data,
        iris_dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris_dataset.target,
    )

    selected_model_name = "knn"
    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
    model_pipeline.fit(train_features, train_labels)
    accuracy_score = evaluate_model(model_pipeline, test_features, test_labels)
    LOGGER.info("%s accuracy: %.4f", selected_model_name, accuracy_score)

    return {
        "model_name": selected_model_name,
        "score": accuracy_score,
        "model_pipeline": model_pipeline,
        "target_names": iris_dataset.target_names,
        "feature_names": list(iris_dataset.feature_names),
    }


def save_model_artifact(training_result: TrainResult, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_data = {
        "model_pipeline": training_result["model_pipeline"],
        "target_names": training_result["target_names"],
        "feature_names": training_result["feature_names"],
        "model_name": training_result["model_name"],
        "accuracy": training_result["score"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Atomic write to avoid partially written artifact files.
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".joblib", dir=artifact_path.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        joblib.dump(artifact_data, temp_path)

    temp_path.replace(artifact_path)
    hash_path = artifact_path.with_suffix(".sha256")
    hash_path.write_text(calculate_file_sha256(artifact_path), encoding="utf-8")
    LOGGER.info("Saved model '%s' to: %s", training_result["model_name"], artifact_path)


def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    configure_logging()
    cli_args = parse_cli_args()
    try:
        training_result = train_knn_model(
            random_state=cli_args.random_state,
            test_size=cli_args.test_size,
            n_neighbors=cli_args.n_neighbors,
        )
        LOGGER.info(
            "Best model: %s (accuracy=%.4f)",
            training_result["model_name"],
            training_result["score"],
        )
        save_model_artifact(training_result, cli_args.artifact_path)
    except Exception:
        LOGGER.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
