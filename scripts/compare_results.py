"""Compare un fichier de résultats reconstruit avec celui du dépôt.

Ignore les durées et les pics de mémoire, qui changent d'une machine à l'autre ;
les nombres sont comparés avec une tolérance relative de 1e-9. Les fichiers
.jsonl sont comparés ligne à ligne, par clé (« cle »).

Usage : python scripts/compare_results.py reports/mesures/test_depot.json reports/mesures/test.json
"""

import json
import math
import sys
from pathlib import Path

TOLERANCE = 1e-9


def ignored(key: str) -> bool:
    """Durées et mémoire : elles dépendent de la machine, pas des résultats."""
    return "duree" in key or "pic_memoire" in key or key.endswith(("_s", "_ms"))


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return {r["cle"]: r for r in map(json.loads, filter(None, text.splitlines()))}
    return json.loads(text)


def differences(a, b, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if ignored(str(key)):
                continue
            if key not in a or key not in b:
                out.append(f"{where}/{key} : présent d'un seul côté")
            else:
                out += differences(a[key], b[key], f"{where}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where} : {len(a)} éléments contre {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{where}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        return [] if same else [f"{where} : {a} contre {b}"]
    return [] if a == b else [f"{where} : {a!r} contre {b!r}"]


def main() -> None:
    depot, nouveau = Path(sys.argv[1]), Path(sys.argv[2])
    ecarts = differences(load(depot), load(nouveau))
    print(f"{nouveau.name} : {len(ecarts)} écart(s) avec {depot.name} (durées et mémoire ignorées)")
    for ligne in ecarts[:50]:
        print("  " + ligne)
    sys.exit(1 if ecarts else 0)


if __name__ == "__main__":
    main()
