import json
import datetime
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(input_features: dict, probability: float, prediction: int):
    """Enregistre un événement de prédiction dans un fichier JSONL."""
    event = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "features": input_features,
        "probability": probability,
        "prediction": prediction
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
