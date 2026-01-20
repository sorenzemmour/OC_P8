import pandas as pd
import streamlit as st
from dashboard.config import get_api_url
from dashboard.api_client import ApiClient
from dashboard.ui_components import render_barh

st.set_page_config(page_title="Explications", layout="wide")
st.title("🧠 Explications du score (SHAP)")

if "current_client" not in st.session_state:
    st.warning("Va d’abord sur l’accueil pour charger un client.")
    st.stop()

api = ApiClient(get_api_url())
features = st.session_state["current_client"]["features"]

top_n = st.slider("Nombre de variables à afficher", min_value=5, max_value=20, value=10, step=1)

if st.button("🔎 Calculer l’explication (explain)"):
    try:
        r = api.explain(features, top_n=top_n)
        if r.status_code == 200:
            st.session_state["last_explain"] = r.json()
            st.success("Explication calculée.")
        else:
            st.error(f"Erreur {r.status_code}: {r.text}")
    except Exception as e:
        st.error(str(e))

if "last_explain" not in st.session_state:
    st.info("Clique sur 'Calculer l’explication'.")
    st.stop()

exp = st.session_state["last_explain"]

local = pd.DataFrame(exp["top_contributions"])
# Pour un tri visuel : importance = abs(shap_value)
local["abs"] = local["shap_value"].abs()
local = local.sort_values("abs", ascending=False).drop(columns=["abs"])

global_imp = pd.DataFrame(exp["global_importance"]).sort_values("importance", ascending=False)

colL, colG = st.columns(2)

with colL:
    render_barh(
        "Importance locale (ce client)",
        local.iloc[::-1],  # invert for barh
        x_col="shap_value",
        y_col="feature",
        caption="Valeurs SHAP : >0 augmente le risque, <0 diminue le risque (échelle du modèle).",
    )

with colG:
    render_barh(
        "Importance globale (modèle)",
        global_imp.head(top_n).iloc[::-1],
        x_col="importance",
        y_col="feature",
        caption="Importance globale = moyenne des |SHAP| sur un échantillon de clients.",
    )

st.markdown("### Résumé accessible")
st.write(
    "Cette page montre deux niveaux d’explication : "
    "**globale** (ce que le modèle regarde le plus en général) et **locale** "
    "(ce qui a le plus pesé dans le cas de ce client)."
)
