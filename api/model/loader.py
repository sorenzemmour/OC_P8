import os
import joblib
import numpy as np

# --------------------------------------------------
# 1) MODE TEST (GitHub Actions)
# --------------------------------------------------
TESTING = os.getenv("TESTING") == "1"

# --------------------------------------------------
# 2) MLFLOW ACTIVÉ UNIQUEMENT SI ENV=1
# (Render utilisera USE_MLFLOW=0 donc MLflow sera ignoré)
# --------------------------------------------------
USE_MLFLOW = os.getenv("USE_MLFLOW", "0") == "1"

if USE_MLFLOW and not TESTING:
    import mlflow
    import mlflow.sklearn

RUN_ID = "220b6b0558b049688b2ece173f794542"
MODEL_URI = f"runs:/{RUN_ID}/model"

# --------------------------------------------------
# 3) Chemin ABSOLU vers le modèle local
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = None


def load_model():
    """
    Charge le modèle utilisé par l'API.
    - En mode TESTING → DummyModel (simple, léger, fiable)
    - En mode normal :
         → Essaye MLflow si USE_MLFLOW=1
         → Sinon charge le modèle local .pkl
    """

    global model

    # Déjà chargé = pas besoin de recharger
    if model is not None:
        return model

    # --------------------------------------------------
    # 🧪 1) MODE TEST → modèle factice
    # --------------------------------------------------
    if TESTING:
        print("🧪 Mode TESTING — utilisation d’un DummyModel.")

        class DummyModel:
            def predict(self, X):
                return [0]

            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])  # probabilité stable

        model = DummyModel()
        return model

    # --------------------------------------------------
    # 🔄 2) MLFLOW (uniquement si activé)
    # --------------------------------------------------
    if USE_MLFLOW:
        try:
            print("🔄 Tentative de chargement via MLflow...")
            model = mlflow.sklearn.load_model(MODEL_URI)
            print("✅ Modèle chargé depuis MLflow.")
            return model
        except Exception as e:
            print(f"⚠️ MLflow indisponible : {e}")
            print("➡️ Fallback vers modèle local.")

    # --------------------------------------------------
    # 📦 3) MODE LOCAL (Render + Local Dev)
    # --------------------------------------------------
    try:
        print(f"🔄 Chargement modèle local : {LOCAL_MODEL_PATH}")

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable : {LOCAL_MODEL_PATH}")

        model = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Modèle local chargé.")
        return model

    except Exception as e:
        print(f"❌ Impossible de charger le modèle local : {e}")
        raise RuntimeError("Aucun modèle disponible pour l'inférence.")
