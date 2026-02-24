from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def test_models_endpoint():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data
    assert len(data["available_models"]) >= 1

def test_predict_endpoint():
    model_id = "best_model_og_LN"

    # create a simple RGB image in memory
    img = Image.new("RGB", (128, 128), (120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("sample.jpg", buf, "image/jpeg")}
    r = client.post(f"/predict?model_id={model_id}", files=files)

    assert r.status_code == 200, r.text
    data = r.json()

    assert data["model_id"] == model_id
    assert data["prediction"] in {"Typical", "Atypical"}

    probs = data["probabilities"]
    assert set(probs.keys()) == {"Typical", "Atypical"}
    assert 0.0 <= probs["Typical"] <= 1.0
    assert 0.0 <= probs["Atypical"] <= 1.0
    assert abs((probs["Typical"] + probs["Atypical"]) - 1.0) < 1e-2
