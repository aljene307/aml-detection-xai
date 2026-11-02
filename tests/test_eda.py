import pandas as pd
import pytest

from aml_xai.data import convert_to_parquet, load_training_period
from aml_xai.eda import (
    amount_shares,
    currency_differs,
    location_differs,
    rate_by,
    rate_by_day_of_month,
    rate_by_hour,
    rate_by_week,
    rate_by_weekday,
    typology_table,
    wilson_interval,
)
from test_data import MINI_CSV


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-10-03 10:00", "2022-10-03 10:30", "2022-10-04 23:00", "2022-10-11 01:00",
                 "2022-10-18 01:00"]
            ),
            "Amount": [10.0, 100.0, 1000.0, 10000.0, 20.0],
            "Payment_type": pd.Categorical(["ACH", "ACH", "Cash", "Cash", "ACH"]),
            "Payment_currency": ["UK pounds", "Euro", "Euro", "Euro", "Euro"],
            "Received_currency": ["UK pounds", "Euro", "Dirham", "Euro", "Euro"],
            "Sender_bank_location": ["UK", "UK", "UK", "Spain", "UK"],
            "Receiver_bank_location": ["UK", "UAE", "UK", "Spain", "UK"],
            "Is_laundering": [0, 1, 1, 0, 0],
            "Laundering_type": pd.Categorical(
                ["Normal_Fan_In", "Cycle", "Cycle", "Normal_Fan_In", "Normal_Fan_In"]
            ),
        }
    )


def test_intervalle_de_wilson():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    low_zero, high_zero = wilson_interval(0, 1000)
    assert low_zero == pytest.approx(0, abs=1e-9)
    assert 0 < high_zero < 0.005


def test_taux_par_categorie(df):
    table = rate_by(df, "Payment_type")
    assert table.loc["ACH", "transactions"] == 3
    assert table.loc["ACH", "blanchiment"] == 1
    assert table.loc["Cash", "taux_pct"] == pytest.approx(50)


def test_taux_par_heure_et_jour(df):
    assert rate_by_hour(df).loc[10, "transactions"] == 2
    days = rate_by_weekday(df)
    assert days.loc["lundi", "transactions"] == 1 + 1
    assert days.loc["mardi", "taux_pct"] == pytest.approx(100 / 3)
    assert rate_by_day_of_month(df).loc[3, "transactions"] == 2


def test_semaines_incompletes_retirees(df):
    table = rate_by_week(df)
    # trois semaines présentes : la première et la dernière sont retirées
    assert len(table) == 1
    assert table.index[0] == pd.Timestamp("2022-10-10")
    assert table["transactions"].iloc[0] == 1


def test_differences_devise_et_pays(df):
    assert currency_differs(df).tolist() == [False, False, True, False, False]
    assert location_differs(df).tolist() == [False, True, False, False, False]


def test_parts_de_montants(df):
    table = amount_shares(df)
    assert table["normal_pct"].sum() == pytest.approx(100)
    assert table["blanchiment_pct"].sum() == pytest.approx(100)


def test_typologies(df):
    table = typology_table(df)
    assert table.index[0] == "Normal_Fan_In"
    assert table.loc["Cycle", "blanchiment"] == 1
    assert table.loc["Cycle", "montant_median"] == pytest.approx(550)


def test_periode_entrainement_ne_contient_pas_le_futur(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text(MINI_CSV, encoding="utf-8")
    path = convert_to_parquet(csv, tmp_path / "mini.parquet")
    train, cutoff = load_training_period(path=path)
    assert (train["timestamp"] < cutoff).all()
    assert len(train) == 2  # int(4 * 0,6) = 2 : les lignes d'indice 0 et 1
