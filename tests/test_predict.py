from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_SAMPLE = {
    "EXT_SOURCE_3": 0.45,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_1": 0.75,
    "REG_CITY_NOT_WORK_CITY": 1,
    "DAYS_ID_PUBLISH": -500,
    "DAYS_LAST_PHONE_CHANGE": -300.5,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "DAYS_EMPLOYED": -2000,
    "DAYS_BIRTH": -12000
}

def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    data = response.json()

    assert "probability_default" in data
    assert "prediction" in data
    assert "threshold_used" in data
    assert "business_cost_FN" in data
    assert "business_cost_FP" in data
