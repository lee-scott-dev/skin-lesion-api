from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import time

from app.model_registry import MODEL_REGISTRY
from app.preprocess import preprocess_pil

app = FastAPI(title="HAM10000 Skin Lesion Inference Service")

_loaded_models: dict[str, tf.keras.Model] = {} # Cache loaded models so don’t reload each request

LABELS = ["Typical", "Atypical"]  # Matches binary classes from models

@app.get("/")
def root():
    return {"status": "ok", "message": "Service running"}

@app.get("/models")
def list_models():
    return {
        "available_models": [
            {"model_id": k, "family": v.family, "notes": v.notes}
            for k, v in MODEL_REGISTRY.items()
        ]
    }

def get_model(model_id: str) -> tf.keras.Model:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    if model_id not in _loaded_models:
        spec = MODEL_REGISTRY[model_id]
        _loaded_models[model_id] = tf.keras.models.load_model(str(spec.path), compile=False)
    return _loaded_models[model_id]

@app.post("/predict")
async def predict(model_id: str, file: UploadFile = File(...)):
    t0 = time.perf_counter()

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t1 = time.perf_counter()
    x = preprocess_pil(image)
    t2 = time.perf_counter()

    model = get_model(model_id)
    t3 = time.perf_counter()

    probs = model.predict(x, verbose=0)[0]
    t4 = time.perf_counter()

    probs = probs.astype(float)

    top_idx = int(np.argmax(probs))
    return {
        "model_id": model_id,
        "prediction": LABELS[top_idx],
        "probabilities": {
            "Typical": float(probs[0]),
            "Atypical": float(probs[1]),
        },
        "latency_ms": {
            "preprocess_ms": (t2 - t1) * 1000,
            "model_fetch_ms": (t3 - t2) * 1000,
            "predict_ms": (t4 - t3) * 1000,
            "total_ms": (t4 - t0) * 1000,
        }
    }