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
    if st.button("🔍 Analyser le dossier client"):
        try:
            # 1) Charger le client
            r_client = client.get_client(int(sk_id))
            if r_client.status_code != 200:
                st.error(f"Erreur chargement client : {r_client.status_code} {r_client.text}")
                st.stop()

            payload = r_client.json()
            st.session_state["current_client"] = payload

            # 2) Calculer le score
            features = payload["features"]
            r_pred = client.predict(features)
            if r_pred.status_code != 200:
                st.error(f"Erreur calcul score : {r_pred.status_code} {r_pred.text}")
                st.stop()

            st.session_state["last_predict"] = r_pred.json()

            st.success("✅ Dossier chargé et score calculé.")

        except Exception as e:
            st.error(f"Erreur inattendue : {e}")


st.markdown("---")
st.write("➡️ Utilise le menu à gauche pour aller sur **Synthèse**, **Explications**, **Comparaisons**, etc.")
st.caption(f"Seuil actuel du modèle : {threshold}")
