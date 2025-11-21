import joblib
import os

MODEL_PATH = "model/model.pkl"
model = None

def load_model():
    global model
    if model is None:
        print("🔄 Chargement du modèle local...")
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé.")
    return model
