import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def risk_label(pred: int) -> str:
    return "Risque élevé (refus)" if pred == 1 else "Risque faible (accord)"

def distance_to_threshold(prob: float, threshold: float) -> float:
    return prob - threshold

def render_risk_bar(prob: float, threshold: float):
    """
    Barre simple et lisible (accessibilité : repères + texte).
    """
    st.write("### Niveau de risque")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Probabilité de défaut: {fmt_pct(prob)} — Seuil: {fmt_pct(threshold)}")

    delta = distance_to_threshold(prob, threshold)
    if delta >= 0:
        st.warning(f"Au-dessus du seuil de {fmt_pct(delta)} → décision: refus")
    else:
        st.info(f"En-dessous du seuil de {fmt_pct(abs(delta))} → décision: accord")

def render_profile_card(profile: dict):
    st.write("### Profil client")
    # affiche valeurs manquantes
    items = []
    for k, v in profile.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = "Non renseigné"
        items.append((k, v))
    df = pd.DataFrame(items, columns=["Champ", "Valeur"])
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_barh(title: str, df, x_col: str, y_col: str, caption: str | None = None):
    """
    Bar chart horizontal (matplotlib) : facile à lire et accessible.
    """
    st.write(f"### {title}")
    fig, ax = plt.subplots()
    ax.barh(df[y_col], df[x_col])
    ax.invert_yaxis()
    ax.set_xlabel(x_col)
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)
    if caption:
        st.caption(caption)
