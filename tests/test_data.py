import pandas as pd
import pytest

from aml_xai.config import FORBIDDEN_FEATURES, PARQUET_PATH, TARGET, TIMESTAMP, TYPOLOGY
from aml_xai.data import convert_to_parquet, load_transactions, quality_report, train_cutoff

MINI_CSV = """Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type
10:35:19,2022-10-07,1,2,100.5,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Cash_Deposits
10:35:20,2022-10-07,1,3,2500.0,UK pounds,Dirham,UK,UAE,Cross-border,1,Structuring
23:59:59,2022-10-07,4,2,10.0,UK pounds,UK pounds,UK,UK,ACH,0,Normal_Fan_In
00:00:01,2022-10-08,4,5,99.99,Euro,Euro,UK,Spain,Cheque,1,Smurfing
"""


@pytest.fixture
def mini_parquet(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text(MINI_CSV, encoding="utf-8")
    # petit bloc pour forcer plusieurs morceaux
    return convert_to_parquet(csv, tmp_path / "mini.parquet", block_size=300)


def test_conversion_types_et_ordre(mini_parquet):
    df = load_transactions(path=mini_parquet)
    assert len(df) == 4
    assert "Date" not in df.columns and "Time" not in df.columns
    assert df[TIMESTAMP].iloc[2] == pd.Timestamp("2022-10-07 23:59:59")
    assert df[TIMESTAMP].iloc[3] == pd.Timestamp("2022-10-08 00:00:01")
    assert df["transaction_id"].tolist() == [0, 1, 2, 3]
    assert isinstance(df["Payment_type"].dtype, pd.CategoricalDtype)
    assert df["Sender_account"].dtype == "int64"


def test_rapport_qualite_detecte_la_fuite(mini_parquet):
    report = quality_report(load_transactions(path=mini_parquet))
    assert report["nb_typologies_normales"] == 2
    assert report["nb_typologies_suspectes"] == 2
    assert report["typologie_determine_etiquette"] is True
    assert report["ordre_chronologique_respecte"] is True


def test_coupure_entrainement():
    ts = pd.Series(pd.date_range("2022-01-01", periods=10, freq="D"))
    assert train_cutoff(ts) == pd.Timestamp("2022-01-07")


def test_colonnes_interdites():
    assert TARGET in FORBIDDEN_FEATURES and TYPOLOGY in FORBIDDEN_FEATURES


@pytest.mark.skipif(not PARQUET_PATH.exists(), reason="jeu complet non téléchargé")
def test_jeu_complet_colonnes():
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(PARQUET_PATH).metadata
    assert meta.num_rows == 9_504_852
