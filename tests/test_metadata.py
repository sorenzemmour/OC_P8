from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_metadata_endpoint():
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()

    assert "threshold_used" in data
    assert "business_cost_FN" in data
    assert "business_cost_FP" in data
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0

    # Vérifie un élément feature minimal
    f0 = data["features"][0]
    assert "name" in f0
    assert "type" in f0
