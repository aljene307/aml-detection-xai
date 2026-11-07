import json

import numpy as np
import pytest
from sklearn.metrics import fbeta_score, matthews_corrcoef

from aml_xai.evaluation import (
    choose_thresholds,
    confusion_metrics,
    cost_curve,
    daily_top_k,
    false_alerts_by_type,
    load_hypotheses,
    novelty_shares,
    typology_recall,
)


def test_hypotheses_follow_the_formulas(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"analystes": 5, "heures_par_jour": 7, "minutes_par_alerte": 15,
                                "cout_horaire_livres": 30}), encoding="utf-8")
    h = load_hypotheses(path)
    assert h["alertes_par_jour"] == 140
    assert h["cout_alerte_livres"] == 7.5


def test_project_hypotheses_file_is_consistent():
    h = load_hypotheses()
    assert h["alertes_par_jour"] == 140 and h["cout_alerte_livres"] == 7.5
    # la charge de travail tient dans la journée
    assert h["alertes_par_jour"] * h["minutes_par_alerte"] / 60 <= h["analystes"] * h["heures_par_jour"]


@pytest.fixture
def small_case():
    # scores triés à la main : 0,9 (cas, 1 000 £), 0,8 (normal), 0,8 (cas, 50 £), 0,3 (cas, 10 £), 0,1 (normal)
    scores = np.array([0.8, 0.1, 0.9, 0.3, 0.8])
    labels = np.array([0, 0, 1, 1, 1])
    amounts = np.array([5.0, 7.0, 1_000.0, 10.0, 50.0])
    return scores, labels, amounts


def test_cost_curve_by_hand(small_case):
    curve = cost_curve(*small_case, alert_cost=20.0)
    assert curve["alertes"].tolist() == [0, 1, 3, 4, 5]  # les deux 0,8 vont ensemble
    assert curve["montant_manque"].tolist() == [1_060.0, 60.0, 10.0, 0.0, 0.0]
    assert curve["cout_total"].tolist() == [1_060.0, 80.0, 70.0, 80.0, 100.0]
    assert curve["cas_detectes"].tolist() == [0, 1, 2, 3, 3]


def test_thresholds_respect_capacity(small_case):
    curve = cost_curve(*small_case, alert_cost=20.0)
    free = choose_thresholds(curve, capacity=10)
    assert free["cout_minimal"]["alertes"] == 3 and free["cout_minimal"]["respecte_capacite"]
    assert free["retenu"]["alertes"] == 3
    tight = choose_thresholds(curve, capacity=2)
    assert not tight["cout_minimal"]["respecte_capacite"]
    assert tight["retenu"]["alertes"] == 1 and tight["retenu"]["seuil"] == 0.9
    assert tight["au_budget"]["alertes"] == 1


def test_confusion_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    labels = (rng.random(5_000) < 0.05).astype(int)
    alerted = (rng.random(5_000) < 0.1) | (labels.astype(bool) & (rng.random(5_000) < 0.5))
    m = confusion_metrics(alerted, labels)
    assert m["f2"] == pytest.approx(fbeta_score(labels, alerted, beta=2))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(labels, alerted))
    assert m["vp"] + m["fp"] + m["fn"] + m["vn"] == 5_000


def test_confusion_metrics_by_hand():
    m = confusion_metrics(np.array([True, True, False, False]), np.array([1, 0, 1, 0]))
    assert (m["precision"], m["rappel"]) == (0.5, 0.5)
    assert m["f2"] == pytest.approx(0.5)
    assert m["mcc"] == pytest.approx(0.0)


def test_daily_top_k():
    scores = np.array([5, 1, 3, 9, 2, 8])
    days = np.array([1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(daily_top_k(scores, days, 2), [True, False, True, True, False, True])


def test_typology_recall_and_false_alerts():
    alerted = np.array([True, False, True, True, False, True])
    labels = np.array([1, 1, 1, 0, 0, 0])
    types = np.array(["Fan_In", "Fan_In", "Cycle", "Normal_A", "Normal_A", "Normal_B"])
    table = typology_recall(alerted, labels, types)
    assert table.loc["Fan_In", "rappel"] == 0.5 and table.loc["Cycle", "rappel"] == 1.0
    assert (table["ic_bas"] <= table["rappel"]).all() and (table["rappel"] <= table["ic_haut"]).all()
    assert false_alerts_by_type(alerted, labels, types).to_dict() == {"Normal_A": 0.5, "Normal_B": 0.5}


def test_novelty_shares_by_hand():
    alerted = np.array([True, True, True, True, False, False])
    labels = np.array([1, 0, 0, 1, 0, 0])
    new = np.array([True, True, True, False, False, False])
    out = novelty_shares(alerted, labels, {"nouveau": new})["nouveau"]
    assert out["part_transactions"] == 0.5 and out["part_alertes"] == 0.75
    assert out["precision_dans_groupe"] == pytest.approx(1 / 3)
    assert out["precision_hors_groupe"] == 1.0
