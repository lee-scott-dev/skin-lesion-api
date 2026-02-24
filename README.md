# HAM10000 Skin Lesion Inference API

FastAPI service for binary skin lesion classification (Typical vs Atypical)
using LeNet5 models trained on HAM10000.

## Run locally

conda activate skin-lesion-api
uvicorn app.main:app --reload

## Run tests

pytest -q

## Docker

docker build -t skin-lesion-api .
docker run -p 8000:8000 skin-lesion-api

## Endpoint

POST /predict?model_id=best_model_og_LN

Form-data:
file: image
