# Iris FastAPI ML Project

This project trains a production-oriented KNN pipeline on the sklearn Iris dataset and serves predictions with FastAPI.

## Project Structure

- `app/main.py`: FastAPI app and routes
- `app/schemas.py`: Request/response schemas
- `app/services/predict.py`: Model loading and prediction logic
- `ml/train.py`: KNN pipeline training and artifact generation
- `ml/evaluate.py`: Evaluation helper
- `artifacts/model.joblib`: Saved model artifact
- `artifacts/model.sha256`: Artifact integrity hash

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python -m ml.train
```

After training, `artifacts/model.joblib` is created.
The script also creates `artifacts/model.sha256` and the API validates artifact integrity before loading.

Train with custom parameters:

```bash
python -m ml.train --n-neighbors 7 --test-size 0.25 --random-state 123
```

Train and save artifact to a custom path:

```bash
python -m ml.train --artifact-path artifacts/model_v2.joblib
```

## Run the API

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Model Reliability Notes

- Training uses an sklearn `Pipeline` (`StandardScaler` + `KNeighborsClassifier`) to keep train and inference transformations consistent.
- API reloads the model automatically when the artifact file changes.
- API verifies `model.joblib` integrity against `model.sha256` on load.

## Testing

```bash
pytest -q
```

Example request body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0
}
```

Metrics response example:

```json
{
  "model_name": "knn",
  "accuracy": 1.0,
  "trained_at": "2026-04-07T15:10:00+00:00"
}
```
