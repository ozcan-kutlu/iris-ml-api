# Iris FastAPI ML Projesi

Bu proje, sklearn Iris veri seti ile production odaklı bir KNN pipeline eğitir ve FastAPI ile tahmin servisi sunar.

## Canlı Demo / API Dokümanı

- Demo linkini buraya ekleyebilirsin: `https://...`
- API dokümanı (`/docs`) production ortamında varsayılan olarak açık bırakılmamalıdır.

## Proje Yapısı

- `app/main.py`: FastAPI uygulaması ve route tanımları
- `app/schemas.py`: Request/response şemaları
- `app/services/predict.py`: Model yükleme ve tahmin mantığı
- `ml/train.py`: KNN pipeline eğitimi ve artifact oluşturma
- `ml/evaluate.py`: Değerlendirme yardımcı fonksiyonu
- `artifacts/model.joblib`: Kaydedilen model artifact'i
- `artifacts/model.sha256`: Artifact bütünlük hash değeri

## Kurulum

```bash
python -m pip install -r requirements.txt
```

## Model Eğitimi

```bash
python -m ml.train
```

Eğitimden sonra `artifacts/model.joblib` oluşur.
Script ayrıca `artifacts/model.sha256` dosyasını da oluşturur ve API yükleme öncesinde artifact bütünlüğünü doğrular.

Özel parametrelerle eğitim:

```bash
python -m ml.train --n-neighbors 7 --test-size 0.25 --random-state 123
```

Artifact'i farklı bir yola kaydetmek için:

```bash
python -m ml.train --artifact-path artifacts/model_v2.joblib
```

## API'yi Çalıştırma

```bash
uvicorn app.main:app --reload
```

## Endpoint'ler

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Model Güvenilirliği Notları

- Eğitim, train ve inference dönüşümlerini tutarlı tutmak için sklearn `Pipeline` (`StandardScaler` + `KNeighborsClassifier`) kullanır.
- Artifact dosyası değiştiğinde API modeli otomatik olarak yeniden yükler.
- API, yükleme sırasında `model.joblib` dosyasını `model.sha256` ile doğrular.

## Test

```bash
pytest -q
```

Örnek request body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Örnek response:

```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0
}
```

Örnek metrics response:

```json
{
  "model_name": "knn",
  "accuracy": 1.0,
  "trained_at": "2026-04-07T15:10:00+00:00"
}
```
