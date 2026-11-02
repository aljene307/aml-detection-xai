"""Nettoie le notebook sur place et refuse de valider s'il contient une trace locale.

Usage : python scripts/clean_notebook.py [chemin du notebook]
Code de sortie 1 si un chemin local ou une adresse mail reste dans le notebook.
"""

import json
import sys
from pathlib import Path

from aml_xai.config import ROOT
from aml_xai.notebook_tools import clean_notebook, find_personal_data

DEFAULT_NOTEBOOK = ROOT / "notebooks" / "projet_aml.ipynb"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    notebook = clean_notebook(json.loads(path.read_text(encoding="utf-8")))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    problems = find_personal_data(notebook)
    for problem in problems:
        print(problem)
    print(f"{path.name} : nettoyé, {len(problems)} trace(s) locale(s) trouvée(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
