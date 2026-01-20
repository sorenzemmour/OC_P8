from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from api.schemas.input_schema import FEATURE_ORDER
from api.model.preprocess import preprocess_X

# IMPORTANT:
# - on importe get_model et get_clients_df depuis api.main
# - ça évite de recharger le modèle/dataset
import api.main as main


N_BACKGROUND = 500
RANDOM_STATE = 42


def _build_background_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construit la matrice background (n, n_features) dans l'ordre FEATURE_ORDER,
    puis applique preprocess_X (imputation/transform) pour être cohérent avec /predict.
    """
    # On ne garde que les colonnes features
    X_df = df[FEATURE_ORDER].copy()

    # On échantillonne pour limiter coût SHAP
    n = min(N_BACKGROUND, len(X_df))
    X_sample = X_df.sample(n=n, random_state=RANDOM_STATE)

    # Conversion -> numpy (shap préfère du numérique)
    X = X_sample.to_numpy()

    # PAS de preprocess_X ici : le Pipeline gère l'imputation
    X = np.array(X)

    # Sécurisation
    X = np.array(X)
    if X.ndim != 2:
        X = X.reshape(n, -1)

    return X


@lru_cache(maxsize=1)
def get_background() -> np.ndarray:
    """
    Charge et met en cache le background (numpy array).
    """
    df = main.get_clients_df()
    # Vérif rapide (si jamais)
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset de référence incomplet, colonnes manquantes: {missing}")

    return _build_background_matrix(df)


@lru_cache(maxsize=1)
def get_explainer():
    """
    Crée et met en cache l'explainer SHAP.
    Supporte Pipeline sklearn (on explique l'estimator final).
    """
    model = main.get_model()
    background = get_background()

    # Si c'est un Pipeline sklearn, on prend l'estimator final
    estimator = model
    if hasattr(model, "steps") and len(model.steps) > 0:
        estimator = model.steps[-1][1]

    name = type(estimator).__name__.lower()

    # XGBoost / LightGBM: TreeExplainer est le plus adapté
    try:
        if "xgb" in name or "xgboost" in name or "lgbm" in name or "lightgbm" in name:
            return shap.TreeExplainer(estimator, data=background, feature_perturbation="interventional")
    except Exception:
        pass

    # Fallback générique : expliquer via une fonction predict_proba si dispo
    if hasattr(model, "predict_proba"):
        def f(X):
            return model.predict_proba(X)[:, 1]
        return shap.Explainer(f, background)

    return shap.Explainer(estimator, background)



def explain_one(X_one: np.ndarray) -> Dict[str, Any]:
    """
    Calcule les SHAP values pour un seul client.
    Entrée: X_one shape (1, n_features) déjà preprocessé.
    Retour: dict contenant base_value + shap_values (liste) + feature_names.
    """
    X_one = np.array(X_one).reshape(1, -1)
    explainer = get_explainer()

    sv = explainer(X_one)

    # shap.Explanation : values shape (1, n_features) ou (n_features,)
    values = sv.values
    base_values = sv.base_values

    # Normalisation en listes JSON-serializable
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            values_1d = values[0].tolist()
        else:
            values_1d = values.tolist()
    else:
        values_1d = list(values)

    if isinstance(base_values, np.ndarray):
        base_value = float(base_values[0]) if base_values.ndim > 0 else float(base_values)
    else:
        base_value = float(base_values)

    return {
        "base_value": base_value,
        "shap_values": values_1d,
        "feature_names": FEATURE_ORDER,
    }


def top_contributions(
    feature_values: Dict[str, Any],
    shap_values: List[float],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Construit une liste triée des contributions locales (top N) :
    - feature
    - value (valeur feature côté client)
    - shap_value (impact)
    - direction (increase_risk/decrease_risk)
    """
    rows = []
    for fname, s in zip(FEATURE_ORDER, shap_values):
        v = feature_values.get(fname, None)
        rows.append(
            {
                "feature": fname,
                "value": v,
                "shap_value": float(s),
                "direction": "increase_risk" if s > 0 else "decrease_risk",
                "abs_shap": abs(float(s)),
            }
        )

    rows.sort(key=lambda r: r["abs_shap"], reverse=True)
    for r in rows:
        r.pop("abs_shap", None)

    return rows[:top_n]

@lru_cache(maxsize=1)
def get_global_importance(top_n: int = 20):
    """
    Importance globale SHAP = mean(|shap|) sur le background.
    Robuste aux différents formats de sv.values (list, (n,p), (n,p,2), (2,n,p), etc.).
    """
    background = get_background()
    explainer = get_explainer()

    sv = explainer(background)

    p = len(FEATURE_ORDER)

    values = sv.values

    # 1) Si shap renvoie une liste (souvent une par classe), on prend la classe 1 si possible
    if isinstance(values, list):
        # list of arrays: [class0_matrix, class1_matrix]
        if len(values) >= 2:
            values = values[1]
        else:
            values = values[0]

    values = np.array(values)

    # 2) Normaliser en matrice (n, p) correspondant à la classe positive
    if values.ndim == 2:
        # (n, p) OK
        M = values

    elif values.ndim == 3:
        # Cas (n, p, 2) : dernière dim = classes
        if values.shape[0] == background.shape[0] and values.shape[1] == p and values.shape[2] >= 2:
            M = values[:, :, 1]

        # Cas (2, n, p) : première dim = classes
        elif values.shape[0] >= 2 and values.shape[1] == background.shape[0] and values.shape[2] == p:
            M = values[1, :, :]

        # Cas rare : (n, 2, p)
        elif values.shape[0] == background.shape[0] and values.shape[1] >= 2 and values.shape[2] == p:
            M = values[:, 1, :]

        else:
            raise ValueError(f"Format SHAP inattendu pour global_importance: values.shape={values.shape}, attendu p={p}")

    else:
        raise ValueError(f"Format SHAP inattendu pour global_importance: values.ndim={values.ndim}, shape={values.shape}")

    mean_abs = np.mean(np.abs(M), axis=0)

    rows = [{"feature": f, "importance": float(imp)} for f, imp in zip(FEATURE_ORDER, mean_abs.tolist())]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:top_n]

