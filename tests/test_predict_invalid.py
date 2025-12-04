from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_invalid_payload():
    wrong_sample = {"DAYS_BIRTH": -15000}

    response = client.post("/predict", json=wrong_sample)

    # On attend une erreur 422 car Pydantic devrait rejeter le payload
    assert response.status_code == 422


