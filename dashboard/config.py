import os
import streamlit as st

def get_api_url() -> str:
    # Streamlit peut lever FileNotFoundError si aucun secrets.toml n'existe
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"])
    except FileNotFoundError:
        pass

    # fallback env var (local) puis default
    return os.getenv("API_URL", "http://127.0.0.1:8000")
