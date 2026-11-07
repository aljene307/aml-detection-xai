import numpy as np
import pytest

from aml_xai.config import ROOT
from aml_xai.frozen import load_frozen, output_path, sha256_file, sha256_npy


def section_29() -> str:
    text = (ROOT / "docs" / "CHOIX.md").read_text(encoding="utf-8")
    start = text.index("## 29. Choix figés avant l'ouverture du test")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def test_frozen_file_matches_the_pushed_section_29():
    section = section_29()
    frozen = load_frozen()
    for digest in [*frozen["configuration"].values(), *frozen["matrices"].values()]:
        assert digest in section
    for name, spec in frozen["modeles"].items():
        assert spec["empreinte_validation"] in section
        assert spec["scores_validation"] in section
        if spec["seuil"] is not None:
            assert repr(spec["seuil"]) in section, name
    assert len(frozen["modeles"]) == 8


def test_npy_fingerprint_equals_the_saved_file(tmp_path):
    array = np.random.default_rng(0).normal(size=1_000)
    np.save(tmp_path / "a.npy", array)
    assert sha256_npy(array) == sha256_file(tmp_path / "a.npy")
    assert sha256_npy(array + 1e-12) != sha256_npy(array)


def test_test_result_is_written_once(tmp_path):
    assert output_path(tmp_path, None).name == "test.json"
    with pytest.raises(RuntimeError):
        output_path(tmp_path, "raison")  # rien à corriger avant la première ouverture
    (tmp_path / "test.json").write_text("{}")
    with pytest.raises(RuntimeError):
        output_path(tmp_path, None)  # relance refusée
    with pytest.raises(ValueError):
        output_path(tmp_path, "  ")  # raison écrite obligatoire
    assert output_path(tmp_path, "erreur de calcul du coût").name == "test_correction_1.json"
    (tmp_path / "test_correction_1.json").write_text("{}")
    assert output_path(tmp_path, "autre erreur").name == "test_correction_2.json"
    assert (tmp_path / "test.json").read_text() == "{}"  # le premier est gardé
