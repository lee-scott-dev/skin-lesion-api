```markdown
# HAM10000 Skin Lesion Inference API

REST API built with FastAPI for binary skin lesion classification (Typical vs Atypical) using LeNet-5 models trained on the HAM10000 dataset.  
This project demonstrates model serving, API design, testing, Dockerisation, and CI.

---

## Features

- FastAPI inference service
- Multiple model selection via `model_id`
- Image upload (multipart/form-data)
- Deterministic preprocessing pipeline
- Pytest test suite
- Docker container support
- GitHub Actions CI pipeline

---

## Models

| Model ID | Description |
|---|---|
| `best_model_og_LN` | LeNet-5, no augmentation |
| `best_model_manual_aug_LN` | LeNet-5, manual augmentation |
| `best_model_gs_aug_LN` | LeNet-5, grid-search augmentation |
| `best_model_gan_aug_LN` | LeNet-5, GAN oversampling |

> EfficientNet models and GAN training artifacts are excluded from the repository due to size.

---

## Installation (Local)

```bash
conda activate skin-lesion-api
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

API available at `http://127.0.0.1:8000/docs`

### Running Tests

```bash
pytest -q
```

---

## Docker

```bash
docker build -t skin-lesion-api .
docker run -p 8000:8000 skin-lesion-api
```

---

## API Endpoints

### `GET /` — Health check
```json
{ "status": "ok", "message": "Service running" }
```

### `GET /models` — List available models
```json
{
  "available_models": [
    {
      "model_id": "best_model_og_LN",
      "family": "LN",
      "notes": "LeNet5, no augmentation"
    }
  ]
}
```

### `POST /predict?model_id=best_model_og_LN` — Predict
Content-Type: `multipart/form-data` · Field: `file` — image

```json
{
  "model_id": "best_model_og_LN",
  "prediction": "Typical",
  "probabilities": {
    "Typical": 0.82,
    "Atypical": 0.18
  }
}
```

---

## Project Structure

```
app/
  main.py            # FastAPI app and inference endpoint
  model_registry.py  # Model metadata and paths
  preprocess.py      # Image preprocessing pipeline
models/              # LeNet model weights (tracked)
tests/               # Pytest test suite
Dockerfile
requirements.txt
```

---

## CI

GitHub Actions runs dependency installation and the full pytest suite on every push.
```