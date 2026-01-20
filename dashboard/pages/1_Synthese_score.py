import streamlit as st
from dashboard.ui_components import risk_label, render_risk_bar, render_profile_card, fmt_pct

st.set_page_config(page_title="Synthèse & score", layout="wide")
st.title("🧾 Synthèse décision & score")

if "current_client" not in st.session_state:
    st.warning("Va d’abord sur l’accueil pour charger un client.")
    st.stop()

client_data = st.session_state["current_client"]
profile = client_data.get("profile", {})
st.write(f"Client **{client_data.get('SK_ID_CURR')}**")

# Afficher profil
render_profile_card(profile)

st.markdown("---")

if "last_predict" not in st.session_state:
    st.info("Calcule d’abord le score via l’accueil (bouton 'Calculer score').")
    st.stop()

pred = st.session_state["last_predict"]
prob = float(pred["probability_default"])
threshold = float(pred["threshold_used"])
decision = int(pred["prediction"])

col1, col2, col3 = st.columns(3)
col1.metric("Probabilité de défaut", fmt_pct(prob))
col2.metric("Décision", risk_label(decision))
col3.metric("Seuil", fmt_pct(threshold))

render_risk_bar(prob, threshold)

st.write("### Interprétation (texte)")
if decision == 1:
    st.write(
        f"Le modèle estime une probabilité de défaut de **{fmt_pct(prob)}**. "
        f"Le seuil est **{fmt_pct(threshold)}** : la demande est **refusée**."
    )
else:
    st.write(
        f"Le modèle estime une probabilité de défaut de **{fmt_pct(prob)}**. "
        f"Le seuil est **{fmt_pct(threshold)}** : la demande est **acceptée**."
    )
