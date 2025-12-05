import numpy as np
import joblib
import os

TESTING = os.getenv("TESTING") == "1"

IMPUTER_PATH = "model/imputer.pkl"

# Charger l'imputer uniquement en mode normal
imputer = None
if not TESTING:
    if os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)
    else:
        print(f"⚠️ Attention : imputer non trouvé à {IMPUTER_PATH}. Aucune imputation ne sera appliquée.")


def preprocess_X(X):
    """
    Applique le préprocessing sur X :
    - En mode TESTING → retourne X brut
    - En mode normal → applique l'imputer pré-entraîné
    """
    # Mode test : pas de preprocessing
    if TESTING:
        return X

    X = np.array(X, dtype=float).reshape(1, -1)

    # Si l’imputer n’a pas été chargé, fallback sans imputation
    if imputer is None:
        return X

    # Appliquer imputation réelle
    return imputer.transform(X)
