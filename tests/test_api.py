import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from aml_xai.api_service import HISTORY_FEATURES, history_errors, rank_text
from aml_xai.config import FEATURES_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def examples(client):
    return [client.get(f"/examples/{e['id']}").json() for e in client.get("/examples").json()]


def doc_examples(client) -> list[tuple[str, dict]]:
    """Les exemples tels qu'affichés dans /docs : lus dans le schéma OpenAPI."""
    spec = client.get("/openapi.json").json()
    out = []
    for path in ("/score", "/explain"):
        content = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]
        out += [(path, ex["value"]) for ex in content["examples"].values()]
    assert len(out) == 4
    return out


@pytest.mark.parametrize("modele", ["lof", "lightgbm"])
def test_every_example_shown_in_docs_returns_200(client, modele):
    for path, body in doc_examples(client):
        response = client.post(f"{path}?modele={modele}", json=body)
        assert response.status_code == 200, (path, response.text)


def test_get_endpoints_return_200(client):
    assert client.get("/health").json()["statut"] == "ok"
    info = client.get("/model/info").json()
    assert info["modele_par_defaut"] == "lof" and set(info["modeles"]) == {"lof", "lightgbm"}
    assert len(info["variables_historique"]) == 13
    assert len(client.get("/examples").json()) == 30


def test_examples_follow_the_validated_rule(examples):
    categories = pd.Series([e["categorie"] for e in examples]).value_counts().to_dict()
    assert categories == {"vrai_cas_alerte": 10, "fausse_alerte": 10, "cas_manque": 5, "normale": 5}
    true_types = [e["typologie"] for e in examples if e["categorie"] == "vrai_cas_alerte"]
    assert len(set(true_types)) == 10


def test_every_example_gives_its_expected_scores(client, examples):
    for e in examples:
        for modele in ("lof", "lightgbm"):
            r = client.post(f"/score?modele={modele}", json=e["entree"]).json()
            assert r["score"] == pytest.approx(e["scores_attendus"][modele], rel=1e-12, abs=1e-12), (e["id"], modele)
            assert r["alerte"] == (r["score"] >= r["seuil"])


def test_default_model_is_lof(client, examples):
    assert client.post("/score", json=examples[0]["entree"]).json()["modele"] == "lof"


def test_lightgbm_explanation_adds_up_and_keeps_positive_codes(client, examples):
    for e in examples[:10]:
        r = client.post("/explain?modele=lightgbm", json=e["entree"]).json()
        x = r["explication"]
        # 13 contributions arrondies à 6 décimales : écart d'arrondi d'au plus 13 × 0,5e-6
        assert x["valeur_de_base"] + sum(x["contributions_groupes"].values()) == pytest.approx(r["score"], abs=1e-5)
        assert len(x["codes_motifs"]) <= 3 and all(c["contribution"] > 0 for c in x["codes_motifs"])
        assert all(c["phrase"] for c in x["codes_motifs"])


def test_lof_explanation_is_precomputed_for_examples_only(client, examples):
    e = examples[0]
    r = client.post("/explain", json=e["entree"]).json()
    assert r["explication"]["codes_motifs"] == e["explication_lof"]["codes_motifs"]
    changed = copy.deepcopy(e["entree"])
    changed["montant"] += 1.0
    r = client.post("/explain", json=changed).json()
    assert r["explication"] is None and "seulement pour les exemples" in r["message"]


@pytest.mark.parametrize("change", [
    {"em_nb_1h": 5, "em_nb_24h": 2},                       # plus d'envois sur 1 h que sur 24 h
    {"em_nb_1h": -1},                                      # compteur négatif
    {"re_sans_historique_30j": 1, "re_nb_24h": 3},         # sans historique mais des réceptions
    {"em_ecart_precedent_s": 40 * 86_400},                 # au-delà du plafond de 30 jours
])
def test_incoherent_history_is_refused(client, examples, change):
    body = copy.deepcopy(examples[0]["entree"])
    body["historique"].update(change)
    assert client.post("/score", json=body).status_code == 422


def test_unknown_example_gives_404(client):
    assert client.get("/examples/31").status_code == 404


def test_rank_follows_the_score(client):
    from api.main import state

    s = state["service"]
    low, high = s.rank("lof", 1.0), s.rank("lof", 3.0)
    assert 0 < high < low <= 1
    assert rank_text(0.0023) == "parmi les 0,23 % les plus suspectes des transactions de validation"
    assert rank_text(0.32) == "parmi les 32 % les plus suspectes des transactions de validation"


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="variables calculées absentes")
def test_real_transactions_pass_the_coherence_check():
    rows = pd.read_parquet(FEATURES_PATH, columns=HISTORY_FEATURES).sample(200_000, random_state=0)
    bad = [i for i, h in enumerate(rows.to_dict("records")) if history_errors(h)]
    assert bad == [], history_errors(rows.iloc[bad[0]].to_dict())
    assert np.isfinite(rows.to_numpy()).all()
