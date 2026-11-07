import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor

from aml_xai.explain import (
    deletion_drop,
    group_matrix,
    kernel_contributions,
    lightgbm_contributions,
    load_codes,
    lof_score,
    phrase,
    random_groups,
    reason_codes,
)
from aml_xai.features import model_feature_names
from aml_xai.supervised import lightgbm_model, ranking_scores


def test_every_variable_belongs_to_exactly_one_code():
    g = group_matrix(load_codes())
    assert g.shape == (23, 13)
    np.testing.assert_array_equal(g.sum(axis=1), np.ones(23))


def test_reason_codes_keep_only_positive_groups():
    names = ["A", "B", "C", "D"]
    assert reason_codes(np.array([0.5, 2.0, -1.0, 1.0]), names) == ["B", "D", "A"]
    assert reason_codes(np.array([0.5, -2.0, -1.0, -0.1]), names) == ["A"]  # moins de 3 : on en affiche moins
    assert reason_codes(np.array([-0.5, -2.0, 0.0, -0.1]), names) == []


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3_000, 5)).astype(np.float32)
    y = (x[:, 0] > 2.0).astype(np.int8)
    return x, y


def test_lightgbm_contributions_add_up_to_the_raw_score(small):
    x, y = small
    model = lightgbm_model().fit(x, y)
    contributions, base = lightgbm_contributions(model, x[:200])
    np.testing.assert_allclose(contributions.sum(axis=1) + base, ranking_scores(model, x[:200]), atol=1e-9)
    # la variable qui fait le signal pousse vers « suspect » pour les cas
    assert contributions[y[:200] == 1, 0].mean() > 0


def test_lof_explanation_points_towards_suspect_for_an_outlier(small):
    x, _ = small
    model = LocalOutlierFactor(novelty=True).fit(x)
    score = lof_score(model)
    outlier = np.array([[0, 0, 9.0, 0, 0]], dtype=np.float32)
    # score inversé : l'aberrant a un score plus haut que les transactions normales
    assert score(outlier)[0] > np.percentile(score(x), 99)
    background = x[:30].astype(np.float64)
    values, base = kernel_contributions(score, outlier, background)
    assert values.shape == (1, 5)
    # contribution positive = pousse vers suspect, et elle vient de la variable aberrante
    assert values[0, 2] > 0 and values[0].argmax() == 2
    assert values.sum() + base == pytest.approx(score(outlier)[0], rel=1e-6)


def test_deletion_drop_restores_the_median():
    score = lambda x: np.asarray(x).sum(axis=1)  # noqa: E731
    g = np.array([[1, 0], [1, 0], [0, 1]])
    x = np.array([5.0, 3.0, 1.0])
    medians = np.zeros(3)
    assert deletion_drop(score, x, [0], medians, g) == 8.0
    assert deletion_drop(score, x, [1], medians, g) == 1.0


def test_random_groups_for_the_second_test_avoid_the_chosen_ones():
    rng = np.random.default_rng(0)
    for _ in range(200):
        drawn = random_groups(rng, 13, [2, 5, 7], among_others=True)
        assert len(drawn) == 3 and not set(drawn) & {2, 5, 7}
    drawn = [random_groups(rng, 13, [2, 5, 7], among_others=False) for _ in range(200)]
    assert any(set(d) & {2, 5, 7} for d in drawn)  # la règle d'origine peut retomber sur les codes


USUAL = {"Amount": 3_000.0, "em_ratio_montant_30j": 1.0, "re_ratio_montant_30j": 1.0, "em_nb_1h": 0.0,
         "em_nb_24h": 1.0, "re_nb_1h": 0.0, "re_nb_24h": 1.0, "em_nb_distincts_24h": 1.0,
         "em_nb_distincts_30j": 5.0, "re_nb_distincts_30j": 6.0, "em_ecart_precedent_s": 7_200.0,
         "re_ecart_precedent_s": 7_200.0}


def test_phrases_give_the_direction_and_the_usual_value():
    row = {"Amount": 1_300_000.0, "em_sans_historique_30j": 0.0, "em_ratio_montant_30j": 25.0,
           "re_sans_historique_30j": 1.0, "re_ratio_montant_30j": 1.0, "em_nb_1h": 14.0, "em_nb_24h": 30.0,
           "em_ecart_precedent_s": 300.0, "re_nb_distincts_30j": 2.0,
           "timestamp": pd.Timestamp("2023-05-07 03:12:00"),
           "Payment_currency": "UK pounds", "Received_currency": "Dirham"}
    assert phrase("MONTANT", row, USUAL) == "Montant de 1 300 000,00 £ (habituellement 3 000,00 £)"
    assert phrase("MONTANT_INHABITUEL_EMETTEUR", row, USUAL) == (
        "25,0 fois le montant moyen envoyé sur 30 jours (habituellement 1,0)")
    assert phrase("MONTANT_INHABITUEL_DESTINATAIRE", row, USUAL).startswith("Aucun montant reçu")
    assert phrase("RAFALE_EMETTEUR", row, USUAL) == (
        "14 envois dans l'heure précédente (habituellement 0), 30 envois sur 24 h (habituellement 1)")
    assert phrase("SOURCES_MULTIPLES_DESTINATAIRE", row, USUAL) == (
        "Seulement 2 émetteurs différents en 30 jours (habituellement 6)")
    assert phrase("REPRISE_OU_NOUVEAU_EMETTEUR", row, USUAL) == (
        "Envoi précédent il y a seulement 5,0 min (habituellement 120 min)")
    assert phrase("REPRISE_OU_NOUVEAU_DESTINATAIRE", row, USUAL) == "Aucune réception dans les 30 jours précédents"
    assert phrase("HORAIRE", row, USUAL) == "Le dimanche à 3 h 12"
    assert phrase("DEVISE", row, USUAL) == "Devise différente : payé en UK pounds, reçu en Dirham"
    row["Received_currency"] = "UK pounds"
    assert phrase("DEVISE", row, USUAL) == "Même devise : payé et reçu en UK pounds"
    row["em_ratio_montant_30j"] = 1.0
    assert phrase("MONTANT_INHABITUEL_EMETTEUR", row, USUAL).endswith("(valeur habituelle)")


def test_every_code_has_a_phrase():
    row = {name: 1.0 for name in model_feature_names()}
    row.update({"Amount": 10.0, "timestamp": pd.Timestamp("2023-05-01"), "Payment_currency": "a",
                "Received_currency": "b", "Sender_bank_location": "c", "Receiver_bank_location": "d",
                "Payment_type": "e", "em_sans_historique_30j": 0.0, "re_sans_historique_30j": 0.0})
    for code in load_codes():
        assert phrase(code, row, USUAL)


def test_counts_agree_in_number_and_say_none():
    row = {"em_nb_distincts_24h": 0.0, "em_nb_distincts_30j": 1.0, "re_nb_distincts_30j": 1.0,
           "re_nb_1h": 0.0, "re_nb_24h": 3.0}
    assert phrase("CONTREPARTIES_MULTIPLES_EMETTEUR", row, USUAL) == (
        "Aucun destinataire différent en 24 h (habituellement 1), "
        "seulement 1 destinataire différent en 30 jours (habituellement 5)")
    assert phrase("SOURCES_MULTIPLES_DESTINATAIRE", row, USUAL) == (
        "Seulement 1 émetteur différent en 30 jours (habituellement 6)")
    assert phrase("RAFALE_DESTINATAIRE", row, USUAL) == (
        "Aucune réception dans l'heure précédente (valeur habituelle), 3 réceptions sur 24 h (habituellement 1)")


def test_kernel_shap_is_reproducible():
    # 12 variables : 4 096 combinaisons, plus que les 500 évaluations, donc KernelSHAP tire au hasard
    x = np.random.default_rng(1).normal(size=(500, 12)).astype(np.float32)
    score = lof_score(LocalOutlierFactor(novelty=True).fit(x))
    background = x[:20].astype(np.float64)
    a, _ = kernel_contributions(score, x[100:103], background, seed=7)
    np.random.seed(123)  # un autre tirage entre les deux appels ne doit rien changer
    np.random.random(10)
    b, _ = kernel_contributions(score, x[100:103], background, seed=7)
    np.testing.assert_array_equal(a, b)
    c, _ = kernel_contributions(score, x[100:103], background, seed=8)
    assert not np.array_equal(a, c)  # le test porte bien sur un tirage


def test_durations_share_the_unit_of_the_shorter_one():
    usual = dict(USUAL, re_ecart_precedent_s=51_890.0)
    row = {"re_sans_historique_30j": 0.0, "re_ecart_precedent_s": 105_000.0}
    assert phrase("REPRISE_OU_NOUVEAU_DESTINATAIRE", row, usual) == (
        "Réception précédente il y a 29 h (habituellement 14 h)")
    row["re_ecart_precedent_s"] = 3 * 86_400.0
    usual["re_ecart_precedent_s"] = 2 * 86_400.0
    assert phrase("REPRISE_OU_NOUVEAU_DESTINATAIRE", row, usual) == (
        "Réception précédente il y a 3,0 jours (habituellement 2,0 jours)")
