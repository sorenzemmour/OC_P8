import mlflow
import mlflow.sklearn
import os

RUN_ID = "220b6b0558b049688b2ece173f794542"
MODEL_URI = f"runs:/{RUN_ID}/model"

model = None

def load_model():
    global model
    if model is None:
        try:
            print("🔄 Chargement du modèle depuis MLflow...")
            model = mlflow.sklearn.load_model(MODEL_URI)
            print("✅ Modèle chargé.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            raise RuntimeError("Impossible de charger le modèle depuis MLflow.")
    return model
