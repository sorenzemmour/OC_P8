import os
import joblib
import numpy as np

# Détection du mode test (GitHub Actions)
TESTING = os.getenv("TESTING") == "1"

LOCAL_MODEL_PATH = "model/model.pkl"

model = None


def load_model():
    """
    Charge le modèle utilisé par l'API.
    - En mode TESTING (GitHub Actions) → DummyModel pour éviter les dépendances lourdes.
    - En production / local → Chargement du modèle .pkl.
    """

    global model

    # Si un modèle est déjà chargé, ne pas recharger
    if model is not None:
        return model

    # 🧪 MODE TEST : renvoie un dummy model simple
    if TESTING:
        print("🧪 Mode TESTING détecté — utilisation d’un modèle factice.")

        class DummyModel:
            def predict(self, X):
                return [0]  # cohérent avec un modèle binaire

            def predict_proba(self, X):
                # Retourne une probabilité stable comme un vrai modèle
                return np.array([[0.7, 0.3]])  

        model = DummyModel()
        return model

    # 🗃️ MODE NORMAL → charger le modèle local
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
