from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import tensorflow as tf
import numpy as np

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
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = preprocess_pil(image)
    model = get_model(model_id)

    probs = model.predict(x, verbose=0)[0]
    probs = probs.astype(float)

    top_idx = int(np.argmax(probs))
    return {
        "model_id": model_id,
        "prediction": LABELS[top_idx],
        "probabilities": {
            "Typical": float(probs[0]),
            "Atypical": float(probs[1]),
        }
    }