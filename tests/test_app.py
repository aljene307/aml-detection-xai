import sys
from pathlib import Path

import pytest
import requests

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import client  # noqa: E402
from donnees import budget_le_plus_proche, hypotheses_derivees  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

UNREACHABLE = "http://127.0.0.1:9"


def test_api_url_falls_back_without_secrets(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    assert client.api_url() == client.DEFAULT_API_URL
    monkeypatch.setenv("API_URL", "http://localhost:8000/")
    assert client.api_url() == "http://localhost:8000"


class Answer:
    def __init__(self, code):
        self.status_code = code


def test_call_retries_once_then_gives_the_waking_message(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError("endormie")

    monkeypatch.setattr(client.requests, "request", fail)
    with pytest.raises(client.ApiUnavailable, match="Réveil de l'API, environ une minute"):
        client.call("GET", "/health")
    assert len(calls) == 2  # un seul nouvel essai


def test_call_waits_through_a_waking_answer(monkeypatch):
    answers = iter([Answer(503), Answer(200)])
    monkeypatch.setattr(client.requests, "request", lambda *a, **k: next(answers))
    assert client.call("GET", "/health").status_code == 200


def test_budget_formulas_and_nearest_budget():
    assert hypotheses_derivees(5, 7, 15, 30) == (140, 7.5)
    assert budget_le_plus_proche([1, 2, 3, 10, 20], 12.7) == 3
    assert budget_le_plus_proche([1, 2, 3], 0.5) == 0


def page_script(module: str, submit: bool = False) -> AppTest:
    script = f"""
import sys
sys.path.insert(0, {str(APP)!r})
import client
client.TIMEOUT_S = 1
client.api_url = lambda: {UNREACHABLE!r}
import {module}
{module}.afficher()
"""
    return AppTest.from_string(script, default_timeout=60)


@pytest.mark.parametrize("module", ["page_exemples", "page_budget", "page_resultats", "page_methode"])
def test_offline_pages_work_without_the_api(module):
    app = page_script(module).run()
    assert not app.exception, app.exception
    assert app.title[0].value


def test_budget_page_reacts_to_the_sliders():
    app = page_script("page_budget").run()
    app.slider[0].set_value(10).run()  # 10 analystes : 280 alertes par jour
    assert not app.exception
    assert any("280" in m.value for m in app.metric)


def test_analyse_page_explains_that_the_api_is_waking():
    app = page_script("page_analyser").run()
    assert not app.exception
    app.button[0].click().run()  # « Analyser »
    assert not app.exception
    assert any("Réveil de l'API" in w.value for w in app.warning)


def test_whole_app_starts_with_navigation(monkeypatch):
    monkeypatch.setenv("API_URL", UNREACHABLE)  # le réveil au démarrage ne doit pas sortir sur le réseau
    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Analyser une transaction"


def test_form_sends_exactly_the_chosen_example():
    """Sans modification, le formulaire envoie l'exemple tel quel (secondes comprises) :
    l'API retrouve alors l'explication de LOF calculée à l'avance."""
    script = f"""
import sys, json
sys.path.insert(0, {str(APP)!r})
import client, page_analyser
envois = []
def espion(method, path, payload=None, params=None):
    envois.append(payload)
    raise client.ApiUnavailable("test")
page_analyser.call = espion
page_analyser.afficher()
import streamlit as st
st.session_state["envoi"] = json.dumps(envois[-1]) if envois else ""
"""
    import json

    app = AppTest.from_string(script, default_timeout=60).run()
    app.button[0].click().run()
    exemple = json.loads((APP.parent / "data" / "demo" / "exemples.json").read_text(encoding="utf-8"))[0]
    assert json.loads(app.session_state["envoi"]) == exemple["entree"]


def test_invalid_date_is_refused_in_the_form():
    app = page_script("page_analyser").run()
    app.text_input[0].set_value("30/07/2023 21:58")  # dans un formulaire : pris en compte à l'envoi
    app.button[0].click().run()
    assert any("AAAA-MM-JJ HH:MM:SS" in e.value for e in app.error)


def test_vertical_lines_sit_at_the_right_place_on_log_axes():
    """Sur un axe logarithmique, Plotly place le trait à la valeur réelle et l'étiquette au logarithme."""
    import math

    from graphiques import comparaison_modeles, courbes_budget
    from donnees import lire

    budget = {"alertes_par_jour": [1, 10, 100, 1000], "series": {"LOF, sans étiquette": {"rappel": [0, .1, .3, .6]}}}
    fig = courbes_budget(budget, ["LOF, sans étiquette"], "rappel", 140, {})
    assert fig.layout.shapes[0].x0 == 140
    assert fig.layout.annotations[0].x == pytest.approx(math.log10(140))
    test = lire("reports/mesures/test.json")
    fig = comparaison_modeles(test, lire("reports/mesures/repetition_validation.json"))
    assert fig.layout.shapes[0].x0 == test["taux"]
    assert fig.layout.annotations[0].x == pytest.approx(math.log10(test["taux"]))


def test_analyse_page_shows_the_whole_rank():
    script = f"""
import sys
sys.path.insert(0, {str(APP)!r})
import page_analyser
page_analyser.resultat({{"modele": "lof", "score": 1.06, "seuil": 1.5, "alerte": False, "rang": 0.32,
    "rang_texte": "parmi les 32 % les plus suspectes des transactions de validation", "valeurs_inconnues": [],
    "explication": None, "message": "explication absente"}})
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert any("parmi les 32 % les plus suspectes des transactions de validation" in m.value for m in app.markdown)
