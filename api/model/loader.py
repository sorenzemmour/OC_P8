import os
import mlflow
import mlflow.sklearn

RUN_ID = "220b6b0558b049688b2ece173f794542"
MODEL_URI = f"runs:/{RUN_ID}/model"

model = None

def load_model():
    global model
    if model is None:

        # ---- NEW : mode test → utiliser modèle mock ----
        if os.environ.get("TESTING") == "1":
            print("⏳ Mode TEST : chargement du mock model")
            from api.model.mock_model import MockModel
            model = MockModel()
            return model

        # ---- Mode normal : charger le modèle MLflow ----
        try:
            print("🔄 Chargement du modèle depuis MLflow...")
            model = mlflow.sklearn.load_model(MODEL_URI)
            print("✅ Modèle chargé.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            raise RuntimeError("Impossible de charger le modèle depuis MLflow.")

    return model
