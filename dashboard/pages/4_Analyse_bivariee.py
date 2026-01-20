import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.config import get_api_url
from dashboard.api_client import ApiClient

st.set_page_config(page_title="Analyse bivariée", layout="wide")
st.title("🔎 Analyse bivariée (2 variables)")

if "current_client" not in st.session_state:
    st.warning("Va d’abord sur l’accueil pour charger un client.")
    st.stop()

api = ApiClient(get_api_url(), timeout=60.0)

meta = st.session_state.get("metadata")
if meta is None:
    r = api.metadata()
    r.raise_for_status()
    meta = r.json()
    st.session_state["metadata"] = meta

features_list = [f["name"] for f in meta["features"]]

# Population sample (cache)
if "population_sample" not in st.session_state:
    with st.spinner("Chargement d’un échantillon population..."):
        r = api.population_sample(n=2000)
        if r.status_code != 200:
            st.error(f"Erreur population_sample: {r.status_code} {r.text}")
            st.stop()
        st.session_state["population_sample"] = r.json()

df = pd.DataFrame(st.session_state["population_sample"]["rows"])

client_id = st.session_state["current_client"]["SK_ID_CURR"]
client_features = st.session_state["current_client"]["features"]

colA, colB = st.columns(2)
with colA:
    x_feat = st.selectbox("Variable X", options=features_list, index=0)
with colB:
    y_feat = st.selectbox("Variable Y", options=features_list, index=min(1, len(features_list)-1))

# Séries population
x_pop = pd.to_numeric(df[x_feat], errors="coerce")
y_pop = pd.to_numeric(df[y_feat], errors="coerce")

# Valeurs client
x_client = client_features.get(x_feat, None)
y_client = client_features.get(y_feat, None)

st.caption(f"Client : {client_id}. Population : {len(df)} clients (échantillon).")

fig, ax = plt.subplots()

# Scatter population (sans dépendre uniquement de la couleur)
ax.scatter(x_pop, y_pop, s=10, alpha=0.5)

# Point client mis en évidence
client_plotted = False
try:
    if x_client is not None and y_client is not None:
        xc = float(x_client)
        yc = float(y_client)
        ax.scatter([xc], [yc], s=80, marker="X")
        client_plotted = True
except Exception:
    client_plotted = False

ax.set_xlabel(x_feat)
ax.set_ylabel(y_feat)
ax.set_title("Nuage de points (population) + client")

st.pyplot(fig, clear_figure=True)

# Résumé accessible (WCAG)
if client_plotted:
    st.write(
        f"Résumé : le client se situe au point **({x_feat}={xc:.3g}, {y_feat}={yc:.3g})**. "
        "Le nuage de points représente l’ensemble des clients de l’échantillon."
    )
else:
    st.write(
        "Résumé : impossible d’afficher le point client (valeurs manquantes ou non numériques). "
        "Le nuage de points représente l’ensemble des clients de l’échantillon."
    )
