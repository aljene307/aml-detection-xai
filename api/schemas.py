"""Données reçues et renvoyées par l'API (docs/CHOIX.md, section 32)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from aml_xai.api_service import history_errors

Modele = Literal["lof", "lightgbm"]


class Historique(BaseModel):
    """Les 13 variables d'historique, calculées par le système de la banque sur `[t − fenêtre ; t[`."""

    em_nb_1h: int = Field(ge=0, description="Envois de l'émetteur dans l'heure précédente")
    em_nb_24h: int = Field(ge=0, description="Envois de l'émetteur dans les 24 h précédentes")
    em_ratio_montant_30j: float = Field(gt=0, description="Montant / montant moyen envoyé sur 30 jours (1 sans historique)")
    em_ecart_precedent_s: float = Field(ge=0, description="Secondes depuis l'envoi précédent, plafonnées à 30 jours")
    em_sans_historique_30j: Literal[0, 1] = Field(description="1 si aucun envoi dans les 30 jours précédents")
    em_nb_distincts_24h: int = Field(ge=0, description="Destinataires différents sur 24 h")
    em_nb_distincts_30j: int = Field(ge=0, description="Destinataires différents sur 30 jours")
    re_nb_1h: int = Field(ge=0, description="Réceptions du destinataire dans l'heure précédente")
    re_nb_24h: int = Field(ge=0, description="Réceptions du destinataire dans les 24 h précédentes")
    re_ratio_montant_30j: float = Field(gt=0, description="Montant / montant moyen reçu sur 30 jours (1 sans historique)")
    re_ecart_precedent_s: float = Field(ge=0, description="Secondes depuis la réception précédente, plafonnées à 30 jours")
    re_sans_historique_30j: Literal[0, 1] = Field(description="1 si aucune réception dans les 30 jours précédents")
    re_nb_distincts_30j: int = Field(ge=0, description="Émetteurs différents sur 30 jours")

    @model_validator(mode="after")
    def coherent(self):
        errors = history_errors(self.model_dump())
        if errors:
            raise ValueError("Historique incohérent : " + " ; ".join(errors))
        return self


class Transaction(BaseModel):
    montant: float = Field(gt=0, description="Montant en livres sterling")
    horodatage: datetime = Field(description="Date et heure de la transaction")
    devise_paiement: str = Field(description="Par exemple « UK pounds »")
    devise_reception: str = Field(description="Par exemple « Albanian lek »")
    pays_emetteur: str = Field(description="Pays de la banque de l'émetteur, par exemple « UK »")
    pays_destinataire: str = Field(description="Pays de la banque du destinataire")
    type_paiement: str = Field(description="Par exemple « Cross-border », « Cash Deposit »")
    historique: Historique


class Score(BaseModel):
    modele: Modele
    score: float = Field(description="LOF : score d'anomalie ; LightGBM : score de risque (log-odds). "
                                     "Plus haut = plus suspect")
    seuil: float = Field(description="Seuil figé sur la validation (docs/CHOIX.md, section 29)")
    alerte: bool
    rang: float = Field(description="Part des transactions de validation au moins aussi suspectes")
    rang_texte: str
    valeurs_inconnues: list[str] = Field(description="Champs dont la valeur n'existait pas à l'entraînement "
                                                     "(codés comme une valeur rare)")


class CodeMotif(BaseModel):
    code: str
    libelle: str
    contribution: float
    phrase: str


class Explication(BaseModel):
    valeur_de_base: float
    contributions_groupes: dict[str, float]
    codes_motifs: list[CodeMotif] = Field(description="Au plus 3 codes, seulement de contribution positive")


class ScoreExplique(Score):
    explication: Explication | None
    message: str | None = None
