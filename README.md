# Iris FastAPI ML Projesi

Bu proje, sklearn Iris veri seti ile production odakli bir KNN pipeline egitir ve FastAPI ile tahmin servisi sunar.

## Canli Demo / API Dokumani

- Demo linkini buraya ekleyebilirsin: `https://...`
- API dokumani (`/docs`) production ortaminda varsayilan olarak acik birakilmamasi onerilir.

## Proje Yapisi

- `app/main.py`: FastAPI uygulamasi ve route tanimlari
- `app/schemas.py`: Request/response semalari
- `app/services/predict.py`: Model yukleme ve tahmin mantigi
- `ml/train.py`: KNN pipeline egitimi ve artifact olusturma
- `ml/evaluate.py`: Degerlendirme yardimci fonksiyonu
- `artifacts/model.joblib`: Kaydedilen model artifact'i
- `artifacts/model.sha256`: Artifact butunluk hash degeri

## Kurulum

```bash
python -m pip install -r requirements.txt
```

## Model Egitimi

```bash
python -m ml.train
```

Egitimden sonra `artifacts/model.joblib` olusur.
Script ayrica `artifacts/model.sha256` dosyasini da olusturur ve API yukleme oncesinde artifact butunlugunu dogrular.

Ozel parametrelerle egitim:

```bash
python -m ml.train --n-neighbors 7 --test-size 0.25 --random-state 123
```

Artifact'i farkli bir yola kaydetmek icin:

```bash
python -m ml.train --artifact-path artifacts/model_v2.joblib
```

## API'yi Calistirma

```bash
uvicorn app.main:app --reload
```

## Endpoint'ler

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Model Guvenilirligi Notlari

- Egitim, train ve inference donusumlerini tutarli tutmak icin sklearn `Pipeline` (`StandardScaler` + `KNeighborsClassifier`) kullanir.
- Artifact dosyasi degistiginde API modeli otomatik olarak yeniden yukler.
- API, yukleme sirasinda `model.joblib` dosyasini `model.sha256` ile dogrular.

## Test

```bash
pytest -q
```

Ornek request body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Ornek response:

```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0
}
```

Ornek metrics response:

```json
{
  "model_name": "knn",
  "accuracy": 1.0,
  "trained_at": "2026-04-07T15:10:00+00:00"
}
```
