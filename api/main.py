"""API de détection explicable du blanchiment (docs/CHOIX.md, section 32).

Modèle par défaut : LOF sans étiquette ; LightGBM réglé en comparaison
(`?modele=lightgbm`). Lancement local :
    uvicorn api.main:app --port 8000
"""

import json
from contextlib import asynccontextmanager

import psutil
from fastapi import Body, FastAPI, HTTPException, Query

from aml_xai.api_service import DEFAULT_MODEL, HISTORY_FEATURES, MODELS, Service, rank_text
from aml_xai.config import ROOT

from .schemas import Explication, Modele, Score, ScoreExplique, Transaction

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fichiers vérifiés et modèles chargés une seule fois, au démarrage."""
    service = Service()
    state["service"] = service
    # Entrées des exemples, normalisées comme une requête, pour retrouver l'explication de LOF
    state["exemples_par_entree"] = {key(Transaction(**e["entree"])): e for e in service.examples}
    yield
    state.clear()


app = FastAPI(
    title="Détection explicable du blanchiment (SAML-D)",
    description="Score, rang et codes motifs pour une transaction. Modèle par défaut : LOF sans étiquette "
                "(aucune étiquette pour apprendre) ; LightGBM réglé en comparaison. Les variables "
                "d'historique sont envoyées par le client.",
    version="1.0.0",
    lifespan=lifespan,
)


def key(transaction: Transaction) -> str:
    return json.dumps(transaction.model_dump(mode="json"), sort_keys=True)


def doc_examples() -> dict:
    """Deux exemples réels pour /docs : un vrai cas alerté et une transaction normale."""
    examples = json.loads((ROOT / "data" / "demo" / "exemples.json").read_text(encoding="utf-8"))
    first = {c: next(e for e in examples if e["categorie"] == c) for c in ("vrai_cas_alerte", "normale")}
    return {
        "vrai_cas": {"summary": f"Vrai cas ({first['vrai_cas_alerte']['typologie']})",
                     "value": first["vrai_cas_alerte"]["entree"]},
        "normale": {"summary": "Transaction normale", "value": first["normale"]["entree"]},
    }


EXAMPLES = doc_examples()
ModeleQuery = Query(DEFAULT_MODEL, description="lof (par défaut, sans étiquette) ou lightgbm (comparaison)")


def scored(transaction: Transaction, modele: str):
    service = state["service"]
    x, row, unknown = service.prepare(transaction.model_dump())
    value = service.score(modele, x)
    share = service.rank(modele, value)
    result = {"modele": modele, "score": value, "seuil": service.threshold(modele),
              "alerte": value >= service.threshold(modele), "rang": share, "rang_texte": rank_text(share),
              "valeurs_inconnues": unknown}
    return result, x, row


@app.get("/health")
def health() -> dict:
    """État de l'API et mémoire utilisée par le processus."""
    return {"statut": "ok", "modeles": list(MODELS),
            "memoire_mo": round(psutil.Process().memory_info().rss / 2**20, 1)}


@app.get("/model/info")
def model_info() -> dict:
    """Modèles, seuils figés, hypothèses, variables attendues, codes motifs et empreintes."""
    service = state["service"]
    test = json.loads((ROOT / "reports" / "mesures" / "test.json").read_text(encoding="utf-8"))["modeles"]
    hypotheses = json.loads((ROOT / "config" / "hypotheses.json").read_text(encoding="utf-8"))
    return {
        "modele_par_defaut": DEFAULT_MODEL,
        "modeles": {
            code: {
                "nom": name,
                "score": "score d'anomalie" if code == "lof" else "score de risque (log-odds)",
                "seuil": service.threshold(code),
                "explication_en_direct": code == "lightgbm",
                "test": {"pr_auc": test[name]["pr_auc"], "precision": test[name]["seuil"]["precision"],
                         "rappel": test[name]["seuil"]["rappel"],
                         "alertes_par_jour": test[name]["seuil"]["alertes_par_jour"]},
            }
            for code, name in MODELS.items()
        },
        "hypotheses": hypotheses,
        "variables_historique": HISTORY_FEATURES,
        "codes_motifs": {c: v["libelle"] for c, v in service.codes.items()},
        "empreintes": service.empreintes,
    }


@app.post("/score", response_model=Score)
def score(transaction: Transaction = Body(openapi_examples=EXAMPLES), modele: Modele = ModeleQuery):
    """Score, alerte au seuil figé et rang par rapport à la validation."""
    return scored(transaction, modele)[0]


@app.post("/explain", response_model=ScoreExplique)
def explain(transaction: Transaction = Body(openapi_examples=EXAMPLES), modele: Modele = ModeleQuery):
    """Score et codes motifs. LightGBM : en direct. LOF : calculés à l'avance pour les 30 exemples."""
    result, x, row = scored(transaction, modele)
    if modele == "lightgbm":
        return {**result, "explication": state["service"].explain_lightgbm(x, row)}
    example = state["exemples_par_entree"].get(key(transaction))
    if example is not None:
        return {**result, "explication": Explication(**example["explication_lof"]),
                "message": f"Explication calculée à l'avance (exemple {example['id']})."}
    return {**result, "explication": None,
            "message": "Explication de LOF disponible seulement pour les exemples (/examples) : son calcul "
                       "prend plusieurs secondes. Utiliser ?modele=lightgbm pour une explication en direct."}


@app.get("/examples")
def examples() -> list[dict]:
    """Les 30 exemples du test : catégorie, typologie, vrai cas ou non."""
    return [{k: e[k] for k in ("id", "categorie", "typologie", "blanchiment", "transaction_id")}
            for e in state["service"].examples]


@app.get("/examples/{example_id}")
def example(example_id: int) -> dict:
    """Entrée prête à envoyer à /score ou /explain, scores attendus et explication de LOF."""
    for e in state["service"].examples:
        if e["id"] == example_id:
            return e
    raise HTTPException(status_code=404, detail=f"Exemple {example_id} inconnu (1 à 30).")
