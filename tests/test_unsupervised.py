import joblib
import numpy as np
import pandas as pd
import pytest

from aml_xai.features import model_feature_names
from aml_xai.matrix import LOG_FEATURES, fit_transform_params, transform
from aml_xai.unsupervised import (
    DEFAULT_SETTINGS,
    GRIDS,
    model_from_settings,
    training_rows,
    METHODS,
    anomaly_scores,
    default_model,
    fit,
    sample_rows,
)


def fake_features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=n).astype(np.float32) for name in model_feature_names()}
    for name in LOG_FEATURES:
        data[name] = rng.exponential(scale=50, size=n).astype(np.float32)
    return pd.DataFrame(data)


def test_log_features_are_the_long_tailed_ones():
    names = model_feature_names()
    assert set(LOG_FEATURES) <= set(names)
    assert len(LOG_FEATURES) == 11
    assert "re_nb_distincts_24h" not in LOG_FEATURES
    for name in ["log_montant", "heure_sin", "jour_semaine", "em_sans_historique_30j", "freq_Payment_type"]:
        assert name not in LOG_FEATURES


def test_transform_standardizes_the_learning_rows():
    df = fake_features(5_000, 0)
    matrix = transform(df, fit_transform_params(df))
    assert matrix.dtype == np.float32 and matrix.shape == (5_000, len(df.columns))
    np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(matrix.std(axis=0), 1, atol=1e-4)


def test_log_then_standardization_by_hand():
    df = fake_features(4, 1)
    name = LOG_FEATURES[0]
    df[name] = np.array([0, 1, 3, 7], dtype=np.float32)
    params = fit_transform_params(df)
    logged = np.log1p([0.0, 1.0, 3.0, 7.0])
    expected = (logged - logged.mean()) / logged.std()
    j = list(df.columns).index(name)
    np.testing.assert_allclose(transform(df, params)[:, j], expected, rtol=1e-6)


def test_parameters_come_from_learning_rows_only():
    train, other = fake_features(1_000, 2), fake_features(1_000, 3)
    params = fit_transform_params(train)
    before = transform(train, params)
    changed = other * 100
    transform(changed, params)
    assert fit_transform_params(train) == params
    np.testing.assert_array_equal(transform(train, params), before)


def test_constant_column_does_not_divide_by_zero():
    df = fake_features(100, 4)
    df["jour_semaine"] = 3.0
    matrix = transform(df, fit_transform_params(df))
    assert np.isfinite(matrix).all()


def test_wrong_column_order_is_refused():
    df = fake_features(100, 5)
    params = fit_transform_params(df)
    with pytest.raises(ValueError):
        transform(df[df.columns[::-1]], params)


def test_sample_rows_is_fixed_sorted_and_without_repeats():
    a, b = sample_rows(10_000, 500, seed=42), sample_rows(10_000, 500, seed=42)
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 500 and (np.diff(a) > 0).all()
    assert not np.array_equal(a, sample_rows(10_000, 500, seed=7))
    np.testing.assert_array_equal(sample_rows(300, 500), np.arange(300))


@pytest.fixture(scope="module")
def small_matrix():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(2_000, 5)).astype(np.float32)
    outlier = np.full((1, 5), 12, dtype=np.float32)
    return np.vstack([normal, outlier])


@pytest.mark.parametrize("method", METHODS)
def test_obvious_outlier_gets_the_highest_score(method, small_matrix):
    model = fit(default_model(method), small_matrix)
    scores = anomaly_scores(model, small_matrix, chunk_rows=300)
    assert scores.shape == (len(small_matrix),)
    assert scores.argmax() == len(small_matrix) - 1


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_same_scores_and_reload(method, small_matrix, tmp_path):
    a = anomaly_scores(fit(default_model(method), small_matrix), small_matrix)
    model = fit(default_model(method), small_matrix)
    np.testing.assert_array_equal(anomaly_scores(model, small_matrix), a)
    joblib.dump(model, tmp_path / "m.joblib")
    np.testing.assert_array_equal(anomaly_scores(joblib.load(tmp_path / "m.joblib"), small_matrix), a)


@pytest.mark.parametrize("method", METHODS)
def test_scores_do_not_depend_on_chunk_size(method, small_matrix):
    model = fit(default_model(method), small_matrix)
    np.testing.assert_allclose(
        anomaly_scores(model, small_matrix, chunk_rows=7), anomaly_scores(model, small_matrix), rtol=1e-12
    )


def test_default_settings_are_the_announced_ones():
    forest = default_model("isolation_forest")
    assert (forest.n_estimators, forest.max_samples, forest.max_features) == (100, "auto", 1.0)
    assert default_model("lof").n_neighbors == 20
    assert default_model("autoencodeur").hidden_layer_sizes == (8,)


def test_column_by_column_build_matches_fit_then_transform():
    from aml_xai.matrix import build_matrix

    df = fake_features(3_000, 6)
    matrix, params = build_matrix(((c, df[c].to_numpy()) for c in df.columns), len(df))
    assert params == fit_transform_params(df)
    np.testing.assert_array_equal(matrix, transform(df, params))
    with pytest.raises(ValueError):
        build_matrix(((c, df[c].to_numpy()) for c in df.columns[::-1]), len(df), params)
    with pytest.raises(ValueError):
        build_matrix(((c, df[c].to_numpy()) for c in df.columns[:-1]), len(df), params)


def test_each_grid_contains_the_default_settings():
    for method in METHODS:
        assert DEFAULT_SETTINGS[method] in GRIDS[method]
    assert [len(GRIDS[m]) for m in METHODS] == [12, 8, 3]


@pytest.mark.parametrize("method", METHODS)
def test_default_settings_give_the_default_model_scores(method, small_matrix):
    settings = dict(DEFAULT_SETTINGS[method])
    if method == "lof":
        settings["echantillon"] = len(small_matrix)
    rows = training_rows(method, len(small_matrix), settings)
    data = small_matrix if rows is None else small_matrix[rows]
    a = anomaly_scores(fit(default_model(method), data), small_matrix)
    b = anomaly_scores(fit(model_from_settings(method, settings), data), small_matrix)
    np.testing.assert_array_equal(a, b)


def test_grid_resumes_after_the_last_written_setting(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_unsupervised.py"
    spec = importlib.util.spec_from_file_location("tune_unsupervised", path)
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    monkeypatch.setattr(tune, "RESULTS_PATH", tmp_path / "grilles.jsonl")
    assert tune.done_results() == {}
    first = {"cle": tune.key("lof", {"n_neighbors": 10, "echantillon": 50_000}, 42), "pr_auc": 0.1}
    tune.append_result(first)
    tune.append_result({"cle": "autre", "pr_auc": 0.2})
    assert tune.done_results() == {first["cle"]: first, "autre": {"cle": "autre", "pr_auc": 0.2}}
    assert first["cle"] == "lof|n_neighbors=10|echantillon=50000|graine=42"


def test_lof_seed_control_lists_eight_new_draws():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_unsupervised.py"
    spec = importlib.util.spec_from_file_location("tune_unsupervised", path)
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    tuned = {"n_neighbors": 10, "echantillon": 100_000}
    results = {
        "a": {"methode": "lof", "graine": 42, "reglage": DEFAULT_SETTINGS["lof"], "pr_auc": 0.03, "cle": "a"},
        "b": {"methode": "lof", "graine": 42, "reglage": tuned, "pr_auc": 0.04, "cle": "b"},
    }
    jobs = list(tune.lof_seed_jobs(results))
    assert len(jobs) == 8
    assert [j[1] for j in jobs] == [DEFAULT_SETTINGS["lof"]] * 4 + [tuned] * 4
    assert {j[2] for j in jobs} == {1, 2, 3, 4}


def test_lof_seed_changes_the_sample():
    settings = {"n_neighbors": 20, "echantillon": 500}
    a = training_rows("lof", 10_000, settings)
    np.testing.assert_array_equal(a, training_rows("lof", 10_000, settings, 42))
    np.testing.assert_array_equal(a, sample_rows(10_000, 500))
    assert not np.array_equal(a, training_rows("lof", 10_000, settings, 1))
    assert training_rows("isolation_forest", 10_000, {}, 1) is None
