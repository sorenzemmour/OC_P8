import os
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas.input_schema import CustomerFeatures, FEATURE_ORDER
from api.model.loader import load_model
from api.utils.business_cost import COST_FN, COST_FP
from api.utils.logging import log_prediction
from api.model.preprocess import preprocess_X
from api.explain.shap_explainer import explain_one, top_contributions, get_global_importance



app = FastAPI(
    title="Credit Scoring API",
    description="API pour prédire le risque client",
    version="1.0.0",
)


# Seuil optimal récupéré depuis MLflow
THRESHOLD = 0.42

# Chemin vers un fichier clients (CSV/Parquet) contenant au minimum SK_ID_CURR + les features
# Possible de le surcharger en prod avec une variable d'env CLIENT_DATA_PATH
CLIENT_DATA_PATH = os.getenv("CLIENT_DATA_PATH", "data/application_test.csv")

# (Optionnel) Liste de champs "profil" à renvoyer en plus (si présents dans le fichier)
# Possible  d'adapter plus tard sans casser l'API.
DEFAULT_PROFILE_COLUMNS = [
    "SK_ID_CURR",
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
]


# ----------------------------
# Helpers: cache model & data
# ----------------------------
@lru_cache(maxsize=1)
def get_model():
    """Charge le modèle 1 seule fois (cache process)."""
    return load_model()


@lru_cache(maxsize=1)
def get_clients_df() -> pd.DataFrame:
    """
    Charge le dataset clients (cache process).
    Attendu : colonne SK_ID_CURR + colonnes features.
    """
    if not os.path.exists(CLIENT_DATA_PATH):
        # On ne bloque pas l'API predict si le fichier n'existe pas,
        # mais /client/{id} renverra une erreur explicite.
        raise FileNotFoundError(
            f"Fichier clients introuvable: {CLIENT_DATA_PATH}. "
            "Définis CLIENT_DATA_PATH (env var) ou ajoute le fichier dans data/."
        )

    if CLIENT_DATA_PATH.lower().endswith(".parquet"):
        df = pd.read_parquet(CLIENT_DATA_PATH)
    else:
        df = pd.read_csv(CLIENT_DATA_PATH)

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Le dataset clients doit contenir la colonne 'SK_ID_CURR'.")

    return df


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    # Pydantic v1 : CustomerFeatures.__fields__
    fields_info = CustomerFeatures.__fields__

    features = []
    for name in FEATURE_ORDER:
        f = fields_info.get(name)
        # type simplifié
        t = "unknown"
        if f is not None:
            type_str = str(getattr(f, "type_", "")).lower()
            if "int" in type_str:
                t = "int"
            elif "float" in type_str:
                t = "float"
            elif "bool" in type_str:
                t = "bool"
            elif "str" in type_str:
                t = "str"

        features.append(
            {
                "name": name,
                "type": t,
                "label": name,   # tu pourras remplacer par des labels “humains” plus tard
                "unit": None,
            }
        )

    return {
        "threshold_used": THRESHOLD,
        "business_cost_FN": COST_FN,
        "business_cost_FP": COST_FP,
        "features": features,
        "client_id_field": "SK_ID_CURR",
    }


@app.get("/client/{sk_id}")
def get_client(sk_id: int):
    def _nan_to_none(v):
        # Convertit NaN (numpy/pandas) -> None pour un JSON propre
        return None if pd.isna(v) else v

    try:
        df = get_clients_df()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sub = df.loc[df["SK_ID_CURR"] == sk_id]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Client SK_ID_CURR={sk_id} introuvable.")

    row = sub.iloc[0].to_dict()

    features = {f: _nan_to_none(row.get(f, None)) for f in FEATURE_ORDER}

    profile = {}
    for col in DEFAULT_PROFILE_COLUMNS:
        if col in row:
            profile[col] = _nan_to_none(row.get(col))

    profile["SK_ID_CURR"] = sk_id

    return {
        "SK_ID_CURR": sk_id,
        "features": features,
        "profile": profile,
    }




@app.post("/predict")
def predict(features: CustomerFeatures):
    # Modèle (cache)
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Construire X dans l'ordre attendu
        d = features.dict()
        X = np.array([[d.get(f) for f in FEATURE_ORDER]])

        # Sécurisation shape
        X = np.array(X).reshape(1, -1)

        # Support dummy model
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        else:
            pred_class = int(model.predict(X)[0])
            proba = float(pred_class)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

    pred = int(proba >= THRESHOLD)

    # Log
    log_prediction(features.dict(), float(proba), pred)

    return {
        "probability_default": float(proba),
        "prediction": pred,
        "threshold_used": THRESHOLD,
        "business_cost_FN": COST_FN,
        "business_cost_FP": COST_FP,
    }

@app.post("/explain")
def explain(features: CustomerFeatures, top_n: int = 10):
    # Modèle (cache)
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Même construction que /predict
        d = features.dict()
        X = np.array([[d.get(f) for f in FEATURE_ORDER]])

        # Sécurisation shape (1, n)
        X = np.array(X).reshape(1, -1)

        # Proba / prédiction
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        else:
            pred_class = int(model.predict(X)[0])
            proba = float(pred_class)

        pred = int(proba >= THRESHOLD)

        # SHAP local
        shap_payload = explain_one(X)  # base_value + shap_values + feature_names
        local_top = top_contributions(d, shap_payload["shap_values"], top_n=top_n)

        #SHAP gloabl
        global_imp = get_global_importance(top_n=20)


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'explication : {e}")

    return {
        "probability_default": proba,
        "prediction": pred,
        "threshold_used": THRESHOLD,
        "business_cost_FN": COST_FN,
        "business_cost_FP": COST_FP,
        "base_value": shap_payload["base_value"],
        "top_contributions": local_top,
        "global_importance": global_imp,

    }
