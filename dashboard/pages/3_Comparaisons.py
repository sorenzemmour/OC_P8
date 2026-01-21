import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.config import get_api_url
from dashboard.api_client import ApiClient

st.set_page_config(page_title="Comparaisons", layout="wide")
st.title("📈 Comparaison client vs population")

if "current_client" not in st.session_state:
    st.warning("Va d’abord sur l’accueil pour charger un client.")
    st.stop()

api = ApiClient(get_api_url(), timeout=300)

# Charger metadata
meta = st.session_state.get("metadata")
if meta is None:
    r = api.metadata()
    r.raise_for_status()
    meta = r.json()
    st.session_state["metadata"] = meta

features_list = [f["name"] for f in meta["features"]]

# Charger sample population (cache)
if "population_sample" not in st.session_state:
    with st.spinner("Chargement d’un échantillon population..."):
        r = api.population_sample(n=2000)
        if r.status_code != 200:
            st.error(f"Erreur population_sample: {r.status_code} {r.text}")
            st.stop()
        st.session_state["population_sample"] = r.json()

pop = st.session_state["population_sample"]
df = pd.DataFrame(pop["rows"])

client_id = st.session_state["current_client"]["SK_ID_CURR"]
client_features = st.session_state["current_client"]["features"]

st.caption(f"Population : échantillon de {len(df)} clients. Client sélectionné : {client_id}")

feat = st.selectbox("Variable à comparer", options=features_list, index=0)

# Valeur client
x_client = client_features.get(feat, None)

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Distribution (population) + position du client")

    # Préparer série population
    s = pd.to_numeric(df[feat], errors="coerce")

    fig, ax = plt.subplots()
    ax.hist(s.dropna(), bins=30)
    ax.set_xlabel(feat)
    ax.set_ylabel("Nombre de clients")

    if x_client is not None:
        try:
            xc = float(x_client)
            ax.axvline(xc, linestyle="--")
            ax.set_title(f"Client: {feat} = {xc}")
        except Exception:
            ax.set_title("Valeur client non numérique (affichée dans le tableau).")
    else:
        ax.set_title("Valeur client : Non renseigné")

    st.pyplot(fig, clear_figure=True)

with col2:
    st.write("### Statistiques (population)")

    if s.dropna().empty:
        st.info("Pas de données numériques suffisantes pour cette variable.")
        st.stop()

    median = float(np.nanmedian(s))
    q1 = float(np.nanpercentile(s, 25))
    q3 = float(np.nanpercentile(s, 75))

    st.metric("Médiane", f"{median:.3g}")
    st.metric("Q1 (25%)", f"{q1:.3g}")
    st.metric("Q3 (75%)", f"{q3:.3g}")

    if x_client is not None:
        try:
            xc = float(x_client)
            pct = float((s.dropna() < xc).mean() * 100.0)
            st.metric("Percentile client", f"{pct:.1f} %")
            st.caption(
                f"Résumé accessible : sur **{feat}**, le client est au **{pct:.1f}e percentile** "
                f"(valeur {xc:.3g})."
            )
        except Exception:
            st.caption("Valeur client non numérique → percentile non calculable.")
    else:
        st.caption("Valeur client manquante → percentile non calculable.")

st.markdown("---")
st.write("### Données (aperçu)")
st.dataframe(df[["SK_ID_CURR", feat]].head(20), use_container_width=True)
