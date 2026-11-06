import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest

from aml_xai.supervised import (
    CURVE_SHARES,
    IMBALANCE_OPTIONS,
    TUNED_GRID,
    account_bootstrap_weights,
    account_subset,
    choose_imbalance,
    lightgbm_model,
    logistic_model,
    positive_weight,
    ranking,
    ranking_scores,
    training_set,
    undersample,
    weighted_average_precision,
)


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_supervised.py"
    spec = importlib.util.spec_from_file_location("train_supervised", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4_000, 5)).astype(np.float32)
    y = ((x[:, 0] > 2.0) | (rng.random(4_000) < 0.002)).astype(np.int8)
    return x, y


def test_undersample_keeps_every_case_and_a_share_of_normals():
    labels = np.array([0] * 1_000 + [1] * 7, dtype=np.int8)
    kept = undersample(labels, share=0.1, seed=3)
    assert (labels[kept] == 1).sum() == 7
    assert (labels[kept] == 0).sum() == 100
    assert (np.diff(kept) > 0).all()
    np.testing.assert_array_equal(kept, undersample(labels, share=0.1, seed=3))
    assert not np.array_equal(kept, undersample(labels, share=0.1, seed=4))


def test_positive_weight_by_hand():
    assert positive_weight(np.array([0, 0, 0, 0, 1, 1])) == 2.0


def test_training_set_options():
    labels = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    assert training_set("aucun", labels) == (None, {})
    assert training_set("poids", labels) == (None, {"scale_pos_weight": 9.0})
    kept, extra = training_set("sous_echantillonnage", labels)
    assert extra == {} and len(kept) == 19
    with pytest.raises(ValueError):
        training_set("smote", labels)


def test_weighted_average_precision_matches_scikit_learn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(5)
    labels = (rng.random(3_000) < 0.05).astype(np.int8)
    scores = np.round(rng.random(3_000) + labels * 0.3, 2)  # avec des ex æquo
    weights = rng.integers(0, 4, 3_000)
    expected = average_precision_score(labels, scores, sample_weight=weights)
    assert weighted_average_precision(labels, weights, ranking(scores)) == pytest.approx(expected, rel=1e-12)
    assert weighted_average_precision(labels, np.ones(3_000), ranking(scores)) == pytest.approx(
        average_precision_score(labels, scores), rel=1e-12)


def test_bootstrap_weights_draw_whole_accounts():
    senders = np.repeat(np.arange(50), 4)
    for weights in account_bootstrap_weights(senders, draws=5, seed=1):
        assert weights.sum() == len(senders)
        assert (weights.reshape(50, 4) == weights.reshape(50, 4)[:, :1]).all()


def bootstrap_case(seed=0):
    rng = np.random.default_rng(seed)
    n = 20_000
    labels = (rng.random(n) < 0.02).astype(np.int8)
    senders = rng.integers(0, 2_000, n)
    noise = rng.random(n)
    return labels, senders, noise


def test_rule_keeps_a_clearly_better_option():
    labels, senders, noise = bootstrap_case()
    scores = {"aucun": noise, "poids": noise + 0.8 * labels, "sous_echantillonnage": noise + 0.8 * labels}
    decision = choose_imbalance(scores, labels, senders, draws=50)
    # sous-échantillonnage identique à poids : jamais strictement meilleur, on garde poids
    assert decision["retenue"] == "poids"
    assert [c["contre"] for c in decision["comparaisons"]] == ["aucun", "poids"]
    assert decision["comparaisons"][0]["part_des_tirages_meilleure"] == 1.0
    assert decision["comparaisons"][1]["part_des_tirages_meilleure"] == 0.0


def test_rule_prefers_the_simplest_option_when_nothing_is_clearly_better():
    labels, senders, noise = bootstrap_case(1)
    same = noise + 0.5 * labels
    decision = choose_imbalance({o: same for o in IMBALANCE_OPTIONS}, labels, senders, draws=50)
    assert decision["retenue"] == "aucun"


def test_rule_compares_with_the_option_retained_so_far():
    labels, senders, noise = bootstrap_case(2)
    scores = {"aucun": noise + 0.5 * labels, "poids": noise, "sous_echantillonnage": noise + 0.9 * labels}
    decision = choose_imbalance(scores, labels, senders, draws=50)
    assert decision["retenue"] == "sous_echantillonnage"
    assert decision["comparaisons"][1]["contre"] == "aucun"


def test_account_subset_is_by_whole_accounts():
    senders = np.repeat(np.arange(200), 5)
    rows = account_subset(senders, 0.1, seed=1)
    chosen = np.unique(senders[rows])
    assert len(chosen) == 20
    assert np.isin(senders, chosen).sum() == len(rows)
    np.testing.assert_array_equal(account_subset(senders, 1.0), np.arange(len(senders)))
    np.testing.assert_array_equal(rows, account_subset(senders, 0.1, seed=1))


def test_lightgbm_defaults_are_the_announced_ones():
    model = lightgbm_model()
    assert (model.n_estimators, model.num_leaves, model.learning_rate) == (100, 31, 0.1)
    assert len(TUNED_GRID) == 6 and CURVE_SHARES[-1] == 1.0


def test_lightgbm_same_seed_same_scores_and_reload(small_data, tmp_path):
    x, y = small_data
    a = lightgbm_model(seed=42).fit(x, y).predict_proba(x)[:, 1]
    model = lightgbm_model(seed=42).fit(x, y)
    np.testing.assert_array_equal(model.predict_proba(x)[:, 1], a)
    joblib.dump(model, tmp_path / "m.joblib")
    np.testing.assert_array_equal(joblib.load(tmp_path / "m.joblib").predict_proba(x)[:, 1], a)


def test_models_learn_the_planted_signal(small_data):
    x, y = small_data
    for model in [lightgbm_model(), logistic_model("aucun", y)]:
        scores = model.fit(x, y).predict_proba(x)[:, 1]
        assert scores[x[:, 0] > 2.0].mean() > 10 * scores[x[:, 0] < 0].mean()


def test_jobs_use_the_option_chosen_by_the_rule():
    script = load_script()
    jobs = list(script.jobs({}, lambda: "poids"))
    phases = [phase for _, phase, _ in jobs]
    assert phases.count("desequilibre") == 3 and phases.count("grille") == 6
    assert phases.count("courbe") == 4 * 5 + 1 and phases[-1] == "logistique"
    assert {job["option"] for _, phase, job in jobs if phase != "desequilibre"} == {"poids"}
    assert len({k for k, _, _ in jobs}) == len(jobs)


def test_resume_reads_written_results(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "RESULTS_PATH", tmp_path / "s.jsonl")
    assert script.done_results() == {}
    script.append_result({"cle": "a", "pr_auc": 0.1})
    assert script.done_results() == {"a": {"cle": "a", "pr_auc": 0.1}}


def test_probabilities_saturate_but_raw_scores_keep_the_order():
    raw = np.array([40.0, 45.0, 3.0])
    proba = 1 / (1 + np.exp(-raw))  # transformation de LightGBM pour la classe 1
    assert proba[0] == proba[1] == 1.0  # ex æquo : l'ordre entre les deux est perdu
    assert raw[1] > raw[0]


def test_ranking_scores_are_the_raw_scores(small_data):
    x, y = small_data
    model = lightgbm_model().fit(x, y)
    raw = ranking_scores(model, x)
    np.testing.assert_array_equal(raw, model.predict(x, raw_score=True))
    np.testing.assert_allclose(1 / (1 + np.exp(-raw)), model.predict_proba(x)[:, 1], rtol=1e-12)
    logistic = logistic_model("aucun", y).fit(x, y)
    np.testing.assert_array_equal(ranking_scores(logistic, x), logistic.decision_function(x))


def test_saturation_check_stops_on_ties():
    script = load_script()
    script.check_no_saturation(np.arange(10_000, dtype=float))
    tied = np.r_[np.full(100, 1e9), np.arange(10_000, dtype=float)]  # 100 ex æquo au maximum
    with pytest.raises(RuntimeError):
        script.check_no_saturation(tied)


def test_early_stopping_runs_with_eval_x_and_eval_y(small_data):
    import lightgbm as lgb

    x, y = small_data
    model = lightgbm_model(None, 42, n_estimators=500, metric="average_precision")
    model.fit(x, y, eval_X=(x,), eval_y=(y,), callbacks=[lgb.early_stopping(10, verbose=False)])
    assert 1 <= model.best_iteration_ <= 500
    assert "average_precision" in model.evals_result_["valid_0"]
