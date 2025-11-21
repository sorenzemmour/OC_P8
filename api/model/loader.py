import os
import joblib
import mlflow
import mlflow.sklearn

RUN_ID = "220b6b0558b049688b2ece173f794542"
MODEL_URI = f"runs:/{RUN_ID}/model"

LOCAL_MODEL_PATH = "model/model.pkl"

model = None

def load_model():
    global model
    if model is not None:
        return model

    # Try MLflow first (local dev environment)
    try:
        print("🔄 Tentative de chargement via MLflow...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ Modèle chargé depuis MLflow.")
        return model
    except Exception as e:
        print(f"⚠️ MLflow indisponible : {e}")

    # Fallback local model
    try:
        print("🔄 Chargement du modèle local...")
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"Fichier {LOCAL_MODEL_PATH} introuvable")

        model = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Modèle local chargé.")
        return model
    except Exception as e:
        print(f"❌ ERREUR — Impossible de charger le modèle local : {e}")
        raise RuntimeError("Aucun modèle disponible pour l'inférence.")
