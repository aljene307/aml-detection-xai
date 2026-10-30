"""Contrôles de qualité du jeu complet, écrits dans reports/qualite_donnees.json.

Usage : python scripts/check_quality.py
"""

import json

from aml_xai.config import REPORTS_DIR
from aml_xai.data import load_transactions, quality_report
from aml_xai.measure import measure


def main() -> None:
    timing = {}
    with measure(timing):
        report = quality_report(load_transactions())
    report["mesure"] = timing
    out = REPORTS_DIR / "qualite_donnees.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
