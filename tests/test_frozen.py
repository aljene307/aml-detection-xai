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


def test_text_fingerprint_ignores_line_endings(tmp_path):
    (tmp_path / "lf.json").write_bytes(b'{\n  "a": 1\n}\n')
    (tmp_path / "crlf.json").write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert sha256_file(tmp_path / "lf.json", normalize_eol=True) == sha256_file(tmp_path / "crlf.json", normalize_eol=True)
    assert sha256_file(tmp_path / "lf.json") != sha256_file(tmp_path / "crlf.json")


def test_compare_results_ignores_durations_and_finds_real_differences(tmp_path):
    import importlib.util
    import json

    path = ROOT / "scripts" / "compare_results.py"
    spec = importlib.util.spec_from_file_location("compare_results", path)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    a = {"pr_auc": 0.5, "duree_s": 10, "pic_memoire_mo": 900, "x": {"notation_s": 3, "rappel": 0.3}}
    b = {"pr_auc": 0.5 + 1e-12, "duree_s": 99, "pic_memoire_mo": 1, "x": {"notation_s": 7, "rappel": 0.3}}
    assert compare.differences(a, b) == []
    b["x"]["rappel"] = 0.31
    assert compare.differences(a, b) == ["/x/rappel : 0.3 contre 0.31"]
    (tmp_path / "a.jsonl").write_text(json.dumps({"cle": "k", "pr_auc": 0.1, "duree_s": 1}) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"cle": "k", "pr_auc": 0.1, "duree_s": 5}) + "\n")
    assert compare.differences(compare.load(tmp_path / "a.jsonl"), compare.load(tmp_path / "b.jsonl")) == []


def test_frozen_text_files_match_with_either_line_ending(tmp_path):
    import hashlib

    from aml_xai.frozen import check_files, matches_frozen

    lf = b'{\n  "a": 1\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    (tmp_path / "f.json").write_bytes(crlf)
    assert matches_frozen(tmp_path / "f.json", hashlib.sha256(lf).hexdigest())   # figé en LF, extrait en CRLF
    (tmp_path / "f.json").write_bytes(lf)
    assert matches_frozen(tmp_path / "f.json", hashlib.sha256(crlf).hexdigest())  # figé en CRLF, extrait en LF
    (tmp_path / "f.json").write_bytes(lf.replace(b"1", b"2"))
    assert not matches_frozen(tmp_path / "f.json", hashlib.sha256(lf).hexdigest())  # contenu changé : refusé
    (tmp_path / "m.npy").write_bytes(crlf)
    assert not matches_frozen(tmp_path / "m.npy", hashlib.sha256(lf).hexdigest())  # binaire : octet par octet
    with pytest.raises(RuntimeError):
        check_files({"f.json": hashlib.sha256(lf).hexdigest()}, tmp_path)


def test_frozen_configuration_still_matches_on_this_pc():
    from aml_xai.frozen import check_files

    assert all(check_files(load_frozen()["configuration"]).values())
