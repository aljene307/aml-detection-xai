import numpy as np
import pandas as pd
import pytest

from aml_xai.config import ACCOUNT_COLUMNS, FORBIDDEN_FEATURES, TARGET, TRANSACTION_ID, TYPOLOGY
from aml_xai.eda import currency_differs, shared_categories
from aml_xai.features import (
    COUNTERPARTY_CANDIDATES,
    DAY_S,
    FREQUENCY_COLUMNS,
    NEUTRAL_RATIO,
    GAP_CAP_S,
    WINDOW_30D_S,
    apply_frequency_maps,
    fit_frequency_maps,
    history_columns,
    history_features,
    history_features_from_parquet,
    model_feature_names,
    transaction_features,
)
from aml_xai.split import assign_periods, period_report, split_cutoffs
from aml_xai.startup import psi, startup_decision, startup_report, with_missing

START = pd.Timestamp("2023-01-01")


def transactions(rows):
    """rows : (secondes depuis START, émetteur, destinataire, montant)."""
    seconds, sender, receiver, amount = zip(*rows)
    return pd.DataFrame(
        {
            "timestamp": START + pd.to_timedelta(seconds, unit="s"),
            "Sender_account": sender,
            "Receiver_account": receiver,
            "Amount": np.array(amount, dtype=float),
        }
    )


def random_transactions(n=3_000, seed=0, n_accounts=40):
    rng = np.random.default_rng(seed)
    # Temps entiers resserrés : beaucoup d'ex æquo et de fenêtres bien remplies
    seconds = np.sort(rng.integers(0, 60 * DAY_S, n))
    seconds[1::7] = seconds[::7][: len(seconds[1::7])]
    seconds = np.sort(seconds)
    return pd.DataFrame(
        {
            "timestamp": START + pd.to_timedelta(seconds, unit="s"),
            "Sender_account": rng.integers(0, n_accounts, n),
            "Receiver_account": rng.integers(0, n_accounts, n),
            "Amount": np.round(rng.uniform(1, 50_000, n), 2),
        }
    )


def brute_force(df, account_column, other_column):
    """Boucle lente et lisible, sans tri ni searchsorted."""
    t = (df["timestamp"] - START).dt.total_seconds().to_numpy().astype(np.int64)
    account = df[account_column].to_numpy()
    other = df[other_column].to_numpy()
    cents = np.rint(df["Amount"].to_numpy() * 100).astype(np.int64)
    out = {name: [] for name in ["nb_1h", "nb_24h", "ratio_montant_30j", "ecart_precedent_s", "sans_historique_30j", "nb_distincts_24h", "nb_distincts_30j", "nb_30j"]}
    for i in range(len(df)):
        same = (account == account[i]) & (t < t[i])
        in_1h = same & (t >= t[i] - 3_600)
        in_24h = same & (t >= t[i] - DAY_S)
        in_30d = same & (t >= t[i] - WINDOW_30D_S)
        out["nb_1h"].append(in_1h.sum())
        out["nb_24h"].append(in_24h.sum())
        out["nb_30j"].append(in_30d.sum())
        out["nb_distincts_24h"].append(len(set(other[in_24h])))
        out["nb_distincts_30j"].append(len(set(other[in_30d])))
        empty = not in_30d.any()
        out["sans_historique_30j"].append(empty)
        ratio = NEUTRAL_RATIO if empty else cents[i] * float(in_30d.sum()) / cents[in_30d].sum()
        out["ratio_montant_30j"].append(ratio)
        out["ecart_precedent_s"].append(min(t[i] - t[same].max(), GAP_CAP_S) if same.any() else GAP_CAP_S)
    return {name: np.array(values, dtype=np.float32) for name, values in out.items()}


def assert_same_arrays(a, b, keep=None):
    assert a.keys() == b.keys()
    for name in a:
        left = a[name] if keep is None else a[name][keep[0]]
        right = b[name] if keep is None else b[name][keep[1]]
        np.testing.assert_array_equal(left, right, err_msg=name)


# ------ Exemple calculé à la main ------


def test_exemple_a_la_main_ex_aequo():
    # Le tableau expliqué : 10:00:00, 10:00:00, 10:30:00, même émetteur
    df = transactions([(36_000, 1, 2, 100), (36_000, 1, 3, 300), (37_800, 1, 4, 600)])
    out = history_features(df)
    assert out["em_nb_1h"].tolist() == [0, 0, 2]
    assert out["em_nb_24h"].tolist() == [0, 0, 2]
    assert out["em_ecart_precedent_s"].tolist() == [GAP_CAP_S, GAP_CAP_S, 1_800]
    assert out["em_sans_historique_30j"].tolist() == [1, 1, 0]
    # Moyenne antérieure = (100 + 300) / 2 = 200, sans la transaction actuelle : 600 / 200 = 3
    assert out["em_ratio_montant_30j"].tolist() == [NEUTRAL_RATIO, NEUTRAL_RATIO, 3]


def test_bornes_de_fenetre():
    # Dernière ligne : fenêtre [1 ; 3 601[, la ligne à 1 (t − 1 h exactement) compte, pas celle à 0
    df = transactions([(0, 1, 2, 10), (1, 1, 2, 10), (3_601, 1, 2, 10)])
    assert history_features(df)["em_nb_1h"].tolist() == [0, 1, 1]
    df = transactions([(0, 1, 2, 10), (DAY_S, 1, 2, 10), (DAY_S + 1, 1, 2, 10)])
    assert history_features(df)["em_nb_24h"].tolist() == [0, 1, 1]


def test_montants_a_nombreuses_decimales():
    # Forme réelle de certaines lignes du jeu : arrondi au centime dans le calcul du rapport
    df = transactions([(0, 1, 2, 10753.26726981541), (10, 1, 2, 10753.27)])
    assert history_features(df)["em_ratio_montant_30j"][1] == 1


def test_ecart_plafonne_a_30_jours():
    # Inactif 40 jours : plafonné à 30 ; 29 jours : vrai écart
    df = transactions([(0, 1, 2, 10), (40 * DAY_S, 1, 2, 500), (69 * DAY_S, 1, 2, 20)])
    out = history_features(df)
    assert out["em_sans_historique_30j"].tolist() == [1, 1, 0]
    # Variables en float32 : 20 / 500 y vaut float32(0,04)
    assert out["em_ratio_montant_30j"].tolist() == [NEUTRAL_RATIO, NEUTRAL_RATIO, np.float32(20 / 500)]
    assert out["em_ecart_precedent_s"].tolist() == [GAP_CAP_S, GAP_CAP_S, 29 * DAY_S]


def test_emetteur_et_destinataire_separes():
    # Le compte 2 reçoit puis envoie : ses deux historiques ne se mélangent pas
    df = transactions([(0, 1, 2, 10), (10, 2, 3, 10), (20, 4, 2, 10)])
    out = history_features(df)
    assert out["em_nb_1h"].tolist() == [0, 0, 0]
    assert out["re_nb_1h"].tolist() == [0, 0, 1]


def test_contreparties_distinctes_a_la_main():
    # 1 écrit à 2, 2, 3 : à la dernière ligne, 3 envois antérieurs vers 2 personnes distinctes
    df = transactions([(0, 1, 2, 10), (10, 1, 2, 10), (20, 1, 3, 10), (30, 1, 4, 10)])
    out = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    assert out["em_nb_24h"].tolist() == [0, 1, 2, 3]
    assert out["em_nb_distincts_24h"].tolist() == [0, 1, 1, 2]
    assert out["em_nb_distincts_30j"].tolist() == [0, 1, 1, 2]
    assert out["em_nb_30j"].tolist() == [0, 1, 2, 3]


# ------ Méthode lente ------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_identique_a_une_boucle_lente(seed):
    df = random_transactions(seed=seed)
    fast = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    for prefix, (account, other) in {
        "em": ACCOUNT_COLUMNS,
        "re": ACCOUNT_COLUMNS[::-1],
    }.items():
        slow = brute_force(df, account, other)
        for name, values in slow.items():
            np.testing.assert_allclose(fast[f"{prefix}_{name}"], values, rtol=1e-6, err_msg=name)


# ------ Anti-fuite : le futur ne change pas le passé ------


@pytest.mark.parametrize("change", ["montants", "contreparties", "ajout", "suppression", "tous_comptes"])
def test_le_futur_ne_change_pas_le_passe(change):
    df = random_transactions()
    cut = START + pd.Timedelta(days=30)
    past = df["timestamp"] < cut
    assert 0.2 < past.mean() < 0.8
    base = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)

    future = df[~past].copy()
    target = future["Sender_account"].iloc[0]
    if change == "montants":
        chosen = future["Sender_account"] == target
        future.loc[chosen, "Amount"] = future.loc[chosen, "Amount"] * 1_000
    elif change == "contreparties":
        future["Receiver_account"] = future["Receiver_account"] + 1_000
    elif change == "ajout":
        future = pd.concat([future, future.assign(Amount=7.0)]).sort_values("timestamp")
    elif change == "suppression":
        future = future[future["Sender_account"] != target]
    else:
        future["Amount"] = 3.5
        future["Sender_account"] = target
    changed = pd.concat([df[past], future], ignore_index=True)
    after = history_features(changed, counterparty_features=COUNTERPARTY_CANDIDATES)

    n_past = int(past.sum())
    # Les lignes du passé ne bougent d'aucun bit (comparaison exacte)
    assert_same_arrays(base, after, keep=(past.to_numpy(), np.arange(len(changed)) < n_past))


def test_controle_du_test_anti_fuite_detecte_une_fuite():
    # Sans le test ci-dessus, rien ne prouve qu'il sait échouer : une variable
    # qui regarde le futur doit être repérée par la même comparaison.
    df = random_transactions()
    past = (df["timestamp"] < START + pd.Timedelta(days=30)).to_numpy()

    def leaky_total(frame):
        return frame.groupby("Sender_account")["Amount"].transform("sum").to_numpy()

    changed = df.copy()
    changed.loc[~past, "Amount"] = changed.loc[~past, "Amount"] * 1_000
    assert not np.array_equal(leaky_total(df)[past], leaky_total(changed)[past])


# ------ Morceaux de comptes ------


@pytest.mark.parametrize("n_chunks", [2, 4, 7])
def test_morceaux_identiques_au_calcul_complet(n_chunks):
    df = random_transactions()
    full = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    chunked = history_features(df, n_chunks=n_chunks, counterparty_features=COUNTERPARTY_CANDIDATES)
    assert_same_arrays(full, chunked)


@pytest.mark.parametrize("n_groups", [1, 2, 8])
def test_lecture_par_groupes_identique_au_calcul_en_memoire(tmp_path, n_groups):
    df = random_transactions().assign(transaction_id=lambda d: np.arange(len(d)))
    path = tmp_path / "mini.parquet"
    df.assign(timestamp=df["timestamp"].astype("datetime64[s]")).to_parquet(path, row_group_size=500)
    full = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    streamed, seen = history_features_from_parquet(
        path, len(df), n_groups, counterparty_features=COUNTERPARTY_CANDIDATES
    )
    assert seen.all()
    assert_same_arrays(full, streamed)


def test_lecture_par_groupes_avec_filtre_de_periode(tmp_path):
    import pyarrow as pa
    import pyarrow.dataset as ds

    df = random_transactions().assign(transaction_id=lambda d: np.arange(len(d)))
    path = tmp_path / "mini.parquet"
    df.assign(timestamp=df["timestamp"].astype("datetime64[s]")).to_parquet(path)
    cut = START + pd.Timedelta(days=30)
    past = (df["timestamp"] < cut).to_numpy()
    streamed, seen = history_features_from_parquet(
        path, len(df), 4, extra_filter=ds.field("timestamp") < pa.scalar(cut, pa.timestamp("s"))
    )
    np.testing.assert_array_equal(seen, past)
    assert_same_arrays(history_features(df), streamed, keep=(past, past))


def test_lecture_par_groupes_refuse_un_nombre_qui_n_est_pas_une_puissance_de_2(tmp_path):
    with pytest.raises(ValueError):
        history_features_from_parquet(tmp_path / "absent.parquet", 1, 6)


def test_ordre_des_lignes_sans_effet():
    df = random_transactions()
    shuffled = df.sample(frac=1, random_state=0)
    base = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    other = history_features(shuffled, counterparty_features=COUNTERPARTY_CANDIDATES)
    assert_same_arrays(base, other, keep=(shuffled.index.to_numpy(), np.arange(len(df))))


# ------ Colonnes interdites ------


def test_colonnes_interdites_absentes_des_variables():
    names = model_feature_names()
    for forbidden in [*FORBIDDEN_FEATURES, *ACCOUNT_COLUMNS, TRANSACTION_ID, "timestamp", "Amount"]:
        assert forbidden not in names
        assert all(forbidden not in name for name in names)
    assert not any("jour_mois" in name for name in names)
    assert TYPOLOGY not in FREQUENCY_COLUMNS


def test_les_etiquettes_ne_changent_aucune_variable():
    # Le calcul fonctionne sans les étiquettes, et leur présence n'y change rien
    df = random_transactions()
    labelled = df.assign(**{TARGET: 1, TYPOLOGY: "Structuring"})
    assert_same_arrays(
        history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES),
        history_features(labelled, counterparty_features=COUNTERPARTY_CANDIDATES),
    )


# ------ Variables de la transaction et codage ------


def test_variables_de_la_transaction():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-02 23:00:00", "2023-01-03 00:00:00", "2023-01-03 12:00:00"]),
            "Amount": [10_000.0, 100.0, 1.0],
            "Payment_currency": pd.Categorical(["UK pounds", "Euro", "Dirham"]),
            "Received_currency": pd.Categorical(["UK pounds", "Dirham", "Dirham"]),
        }
    )
    out = transaction_features(df)
    assert out["log_montant"].tolist() == [4, 2, 0]
    assert out["jour_semaine"].tolist() == [0, 1, 1]  # lundi = 0
    assert out["devises_differentes"].tolist() == [0, 1, 0]
    near = np.hypot(out["heure_sin"][0] - out["heure_sin"][1], out["heure_cos"][0] - out["heure_cos"][1])
    far = np.hypot(out["heure_sin"][1] - out["heure_sin"][2], out["heure_cos"][1] - out["heure_cos"][2])
    assert near < 0.3 < far  # 23 h est proche de 0 h, loin de midi


def test_devises_meme_liste_de_categories_dans_le_meme_ordre():
    # Listes d'origine différentes en contenu et en ordre : les codes bruts ne se comparent pas
    sent = pd.Series(pd.Categorical(["Euro", "Dirham", "Euro"], categories=["Euro", "Dirham"]))
    received = pd.Series(pd.Categorical(["Euro", "Dirham", "Yen"], categories=["Yen", "Dirham", "Euro"]))
    a, b = shared_categories(sent, received)
    assert list(a.cat.categories) == list(b.cat.categories) == ["Dirham", "Euro", "Yen"]
    assert a.astype(str).tolist() == sent.astype(str).tolist()
    assert b.astype(str).tolist() == received.astype(str).tolist()
    df = pd.DataFrame({"Payment_currency": sent, "Received_currency": received})
    assert currency_differs(df).tolist() == [False, False, True]


def test_frequences_apprises_sur_l_apprentissage_seulement():
    train = pd.DataFrame({"Payment_type": pd.Categorical(["ACH"] * 3 + ["Cheque"])})
    other = pd.DataFrame({"Payment_type": pd.Categorical(["Cheque", "Cross-border"])})
    maps = fit_frequency_maps(train, ["Payment_type"])
    assert maps == {"Payment_type": {"ACH": 0.75, "Cheque": 0.25}}
    # Une valeur inconnue vaut 0, et la validation ne change pas ce qui est appris
    assert apply_frequency_maps(other, maps)["freq_Payment_type"].tolist() == [0.25, 0.0]


# ------ Séparation ------


def test_separation_chronologique_sans_seconde_coupee():
    ts = pd.Series(pd.to_datetime(["2023-01-01"] * 5 + ["2023-01-02"] * 3 + ["2023-01-03"] * 2))
    cutoffs = split_cutoffs(ts)
    periods = assign_periods(ts, cutoffs)
    assert periods.tolist() == [0] * 5 + [1] * 3 + [2] * 2
    # Une même seconde ne tombe jamais dans deux périodes
    assert pd.Series(periods).groupby(ts).nunique().max() == 1


def test_rapport_des_periodes():
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "Sender_account": np.arange(n) % 5,
            "Receiver_account": np.arange(n),
            TARGET: (np.arange(n) % 10 == 0).astype(np.int8),
            TYPOLOGY: pd.Categorical(["Structuring" if i % 10 == 0 else "Normal_Fan_In" for i in range(n)]),
        }
    )
    report = period_report(df)
    assert [p["lignes"] for p in report["periodes"].values()] == [60, 20, 20]
    assert report["cas_par_typologie"] == {"Structuring": {"entrainement": 6, "validation": 2, "test": 2}}
    assert report["periodes"]["test"]["lignes_emetteur_connu_entrainement_pct"] == 100
    assert report["periodes"]["test"]["lignes_destinataire_connu_entrainement_pct"] == 0


# ------ Effet de démarrage ------


def test_psi():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=100_000)
    assert psi(reference, reference) == 0
    assert psi(reference, rng.normal(size=100_000)) < 0.01
    assert psi(reference, rng.normal(2, 1, size=100_000)) > 0.25
    with_gaps = np.where(rng.random(100_000) < 0.5, np.nan, reference)
    assert psi(reference, with_gaps) > 0.25  # l'absence de valeur forme sa propre classe


def test_with_missing_restaure_les_absences():
    out = with_missing(
        {
            "em_sans_historique_30j": np.array([1.0, 0.0]),
            "em_ratio_montant_30j": np.array([1.0, 1.0]),
            "em_ecart_precedent_s": np.array([float(GAP_CAP_S), 5.0]),
        }
    )
    assert np.isnan(out["em_ratio_montant_30j"][0]) and out["em_ratio_montant_30j"][1] == 1
    assert out["em_ecart_precedent_s"].tolist() == [GAP_CAP_S, 5]  # le plafond reste une mesure


def decision_for(slices, control):
    return startup_decision({"psi": {"v": slices}, "psi_temoins_max": {"v": control}})


def test_regle_de_decision_du_demarrage():
    # Aucune tranche forte : on n'exclut rien
    assert decision_for({"0-7": 0.2, "7-14": 0.1}, 0.05)["jour_regle_psi"] == 0
    # Forte et au-dessus des témoins : exclue jusqu'à la fin de la dernière tranche différente
    decision = decision_for({"0-7": 0.9, "7-14": 0.3, "14-21": 0.1, "21-30": 0.1}, 0.05)
    assert decision["jour_regle_psi"] == 14
    assert decision["tranches_differentes"] == {"0-7": ["v"], "7-14": ["v"]}
    # Forte, mais pas plus que la variation normale des témoins : pas d'exclusion
    assert decision_for({"0-7": 0.3}, 0.4)["jour_regle_psi"] == 0


def test_demarrage_jamais_avant_la_plus_longue_fenetre():
    # Règle PSI à 0 ou 14 jours : décision retenue à 30 jours, la plus longue fenêtre
    assert decision_for({"0-7": 0.2}, 0.05)["exclure_avant_jour"] == 30
    assert decision_for({"0-7": 0.9, "7-14": 0.9}, 0.05)["exclure_avant_jour"] == 30
    # Une règle PSI plus tardive l'emporterait
    assert decision_for({"0-7": 0.9, "60-67": 0.9}, 0.05)["exclure_avant_jour"] == 67


def test_rapport_de_demarrage_sans_etiquette():
    rng = np.random.default_rng(0)
    days = rng.uniform(0, 100, 200_000)
    stable = rng.normal(size=len(days))
    shifted = stable + np.where(days < 14, 5.0, 0.0)
    report = startup_report(days, {"stable": stable, "decalee": shifted})
    assert report["nb_temoins"] == 9  # débuts 30, 37, …, 86 ; le suivant finirait après le jour 99
    decision = startup_decision(report)
    assert decision["jour_regle_psi"] == 14
    assert set(decision["tranches_differentes"]) == {"0-7", "7-14"}
    assert "stable" not in decision["tranches_differentes"]["0-7"]


def test_noms_de_colonnes_des_deux_cotes():
    assert "em_nb_24h" in history_columns() and "re_nb_24h" in history_columns()
    # Décision de la section 20 : une contrepartie retenue d'un côté seulement
    assert {"em_nb_distincts_24h", "em_nb_distincts_30j", "re_nb_distincts_30j"} <= set(history_columns())
    assert "re_nb_distincts_24h" not in history_columns()
    assert not any(name.endswith("nb_30j") for name in history_columns())


def test_contreparties_retenues_identiques_aux_candidates():
    df = random_transactions()
    kept = history_features(df)
    candidates = history_features(df, counterparty_features=COUNTERPARTY_CANDIDATES)
    assert set(kept) == set(history_columns())
    for name, values in kept.items():
        np.testing.assert_array_equal(values, candidates[name], err_msg=name)
