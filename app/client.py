"""Appels à l'API (seule la page « Analyser une transaction » s'en sert).

Adresse : `st.secrets["API_URL"]` s'il existe, sinon la variable
d'environnement API_URL, sinon l'adresse Render. Délai de 90 s pour laisser
l'API se réveiller, un seul nouvel essai, puis un message clair.
"""

import os
import threading

import requests
import streamlit as st

DEFAULT_API_URL = "https://aml-detection-xai-api.onrender.com"
TIMEOUT_S = 90
# Codes renvoyés par Render pendant le réveil d'un service gratuit
WAKING_CODES = {502, 503, 504}
WAKING_MESSAGE = "Réveil de l'API, environ une minute : réessayez dans un instant."


class ApiUnavailable(Exception):
    """L'API ne répond pas (en veille ou injoignable)."""


def api_url() -> str:
    """Secrets de Streamlit, puis API_URL, puis l'adresse Render par défaut."""
    try:
        url = st.secrets["API_URL"]
    except Exception:  # pas de fichier de secrets, ou pas de clé API_URL
        url = os.environ.get("API_URL") or DEFAULT_API_URL
    return str(url).rstrip("/")


def call(method: str, path: str, payload: dict | None = None, params: dict | None = None) -> requests.Response:
    """Appel avec un seul nouvel essai si l'API dort ; lève ApiUnavailable sinon."""
    for _ in range(2):
        try:
            response = requests.request(method, api_url() + path, json=payload, params=params, timeout=TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError):
            continue
        if response.status_code in WAKING_CODES:
            continue
        return response
    raise ApiUnavailable(WAKING_MESSAGE)


def _ping() -> None:
    try:
        requests.get(api_url() + "/health", timeout=TIMEOUT_S)
    except requests.RequestException:
        pass  # le réveil n'est qu'une avance : la page gère l'absence de réponse


@st.cache_resource(show_spinner=False)
def warm_up() -> bool:
    """Réveille l'API à l'avance, une fois par serveur, sans bloquer l'affichage."""
    threading.Thread(target=_ping, daemon=True).start()
    return True
