import os
import json
from api.utils.logging import log_prediction

def test_log_prediction():
    test_features = {"a": 1}
    probability = 0.5
    prediction = 1

    log_prediction(test_features, probability, prediction)

    assert os.path.exists("logs/predictions.jsonl")

    with open("logs/predictions.jsonl", "r") as f:
        last = f.readlines()[-1]
        data = json.loads(last)

        assert data["features"] == test_features
        assert data["probability"] == probability
        assert data["prediction"] == prediction
