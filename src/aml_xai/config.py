"""Constantes du projet : chemins, colonnes et paramètres de séparation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MEASURES_DIR = REPORTS_DIR / "mesures"

# Jeu SAML-D sur Kaggle (public, téléchargement sans jeton)
KAGGLE_HANDLE = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_HANDLE}"
CSV_PATH = RAW_DIR / "SAML-D.csv"
PARQUET_PATH = PROCESSED_DIR / "saml_d.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
CANDIDATES_PATH = PROCESSED_DIR / "features_candidats.parquet"
MODELS_DIR = ROOT / "models"
FREQUENCY_MAPS_PATH = MODELS_DIR / "frequences.json"
EXPECTED_CSV_BYTES = 996_168_850
EXPECTED_ROWS = 9_504_852

# Colonnes
TARGET = "Is_laundering"
TYPOLOGY = "Laundering_type"
TIMESTAMP = "timestamp"  # fusion de Date et Time
TRANSACTION_ID = "transaction_id"  # ordre d'origine des lignes
ACCOUNT_COLUMNS = ["Sender_account", "Receiver_account"]
CATEGORICAL_COLUMNS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    TYPOLOGY,
]
# Ne doivent jamais entrer dans un modèle : l'étiquette et la typologie,
# dont les valeurs « Normal_… » révèlent l'étiquette.
FORBIDDEN_FEATURES = [TARGET, TYPOLOGY]

# Séparation chronologique 60 / 20 / 20
TRAIN_FRAC = 0.6
VALID_FRAC = 0.2

SEED = 42
