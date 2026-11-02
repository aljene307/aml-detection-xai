"""Nettoyage du notebook avant commit : garder les sorties, pas les traces locales."""

import re

KEPT_NOTEBOOK_METADATA = ("kernelspec", "language_info")
PERSONAL_DATA = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]"  # chemin Windows
    r"|/(?:home|Users)/[\w.-]+/"  # chemin Linux ou macOS
    r"|[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.-]+"  # adresse mail
)
TEXT_MIME_TYPES = ("text/plain", "text/html", "text/markdown")


def clean_notebook(notebook: dict) -> dict:
    """Retire les horodatages et métadonnées d'exécution. Les sorties restent."""
    notebook["metadata"] = {
        key: value for key, value in notebook["metadata"].items() if key in KEPT_NOTEBOOK_METADATA
    }
    notebook["metadata"].get("kernelspec", {})["display_name"] = "Python 3"
    for cell in notebook["cells"]:
        for key in ("execution", "collapsed", "scrolled", "ExecuteTime"):
            cell["metadata"].pop(key, None)
    return notebook


def _as_text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def find_personal_data(notebook: dict) -> list[str]:
    """Liste les chemins locaux et adresses mail du code et des sorties texte."""
    found = []
    for index, cell in enumerate(notebook["cells"]):
        texts = [_as_text(cell["source"])]
        for output in cell.get("outputs", []):
            if "text" in output:
                texts.append(_as_text(output["text"]))
            if "traceback" in output:
                texts.append(_as_text(output["traceback"]))
            for mime in TEXT_MIME_TYPES:
                if mime in output.get("data", {}):
                    texts.append(_as_text(output["data"][mime]))
        for text in texts:
            found += [f"cellule {index} : {m.group()}" for m in PERSONAL_DATA.finditer(text)]
    return found
