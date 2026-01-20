import streamlit as st
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dashboard.config import get_api_url
from dashboard.api_client import ApiClient

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

st.title("📊 Dashboard Crédit — Prêt à dépenser")
st.write("Recherche d’un client et préparation du dossier pour l’analyse.")

api_url = get_api_url()
client = ApiClient(api_url)

# Test connexion
with st.sidebar:
    st.header("Connexion API")
    st.write(f"API_URL = {api_url}")
    if st.button("Tester /health"):
        try:
            r = client.health()
            st.write(r.status_code, r.json())
        except Exception as e:
            st.error(str(e))

# Charger metadata une fois
if "metadata" not in st.session_state:
    try:
        r = client.metadata()
        r.raise_for_status()
        st.session_state["metadata"] = r.json()
    except Exception as e:
        st.error(f"Impossible de charger /metadata : {e}")
        st.stop()

threshold = st.session_state["metadata"]["threshold_used"]

st.subheader("🔎 Recherche client (SK_ID_CURR)")
sk_id = st.number_input("Identifiant client (SK_ID_CURR)", min_value=0, step=1, value=0)

colA, colB = st.columns(2)
with colA:
    if st.button("📥 Récupérer le dossier"):
        try:
            r = client.get_client(int(sk_id))
            if r.status_code == 200:
                payload = r.json()
                st.session_state["current_client"] = payload
                st.success("Dossier client chargé.")
            else:
                st.error(f"Erreur {r.status_code} : {r.text}")
        except Exception as e:
            st.error(str(e))

with colB:
    if st.button("🧮 Calculer score (predict)"):
        if "current_client" not in st.session_state:
            st.warning("Charge d’abord un client avec 'Récupérer le dossier'.")
        else:
            features = st.session_state["current_client"]["features"]
            try:
                r = client.predict(features)
                if r.status_code == 200:
                    st.session_state["last_predict"] = r.json()
                    st.success("Score calculé.")
                else:
                    st.error(f"Erreur {r.status_code} : {r.text}")
            except Exception as e:
                st.error(str(e))

st.markdown("---")
st.write("➡️ Utilise le menu à gauche pour aller sur **Synthèse**, **Explications**, **Comparaisons**, etc.")
st.caption(f"Seuil actuel du modèle : {threshold}")
