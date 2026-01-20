import pandas as pd
from fastapi.testclient import TestClient
import api.main as main

client = TestClient(main.app)

def test_client_endpoint_ok(monkeypatch):
    # Fake dataset minimal
    df = pd.DataFrame([{
        "SK_ID_CURR": 123456,
        "EXT_SOURCE_1": 0.1,
        "EXT_SOURCE_2": 0.2,
        "EXT_SOURCE_3": 0.3,
        "REG_CITY_NOT_WORK_CITY": 1,
        "DAYS_ID_PUBLISH": -100,
        "DAYS_LAST_PHONE_CHANGE": -50,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "DAYS_EMPLOYED": -1000,
        "DAYS_BIRTH": -10000,
        "AMT_INCOME_TOTAL": 50000
    }])

    # Patch la fonction cached qui lit le fichier
    monkeypatch.setattr(main, "get_clients_df", lambda: df)

    r = client.get("/client/123456")
    assert r.status_code == 200
    data = r.json()

    assert data["SK_ID_CURR"] == 123456
    assert "features" in data
    assert "profile" in data
    assert data["profile"]["SK_ID_CURR"] == 123456

def test_client_endpoint_404(monkeypatch):
    df = pd.DataFrame([{"SK_ID_CURR": 1}])
    monkeypatch.setattr(main, "get_clients_df", lambda: df)

    r = client.get("/client/999999")
    assert r.status_code == 404
