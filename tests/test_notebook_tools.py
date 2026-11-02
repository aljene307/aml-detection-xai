from aml_xai.notebook_tools import clean_notebook, find_personal_data


def _notebook(output_text="ok"):
    return {
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": ".venv", "language": "python"},
            "language_info": {"name": "python"},
            "vscode": {"interpreter": {"hash": "abc"}},
        },
        "cells": [
            {
                "cell_type": "code",
                "source": ["print('bonjour')"],
                "metadata": {"execution": {"iopub.execute_input": "2026-11-02T10:00:00Z"}},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": [output_text]}],
            }
        ],
    }


def test_retire_les_horodatages_et_garde_les_sorties():
    notebook = clean_notebook(_notebook())
    assert notebook["cells"][0]["metadata"] == {}
    assert notebook["cells"][0]["outputs"][0]["text"] == ["ok"]
    assert "vscode" not in notebook["metadata"]
    assert notebook["metadata"]["kernelspec"]["display_name"] == "Python 3"


def test_detecte_chemins_et_adresses():
    assert find_personal_data(_notebook("erreur dans C:\\Users\\moi\\projet\\a.py"))
    assert find_personal_data(_notebook("/home/moi/projet/a.py"))
    assert find_personal_data(_notebook("contact : moi@exemple.tn"))


def test_accepte_une_sortie_propre():
    assert find_personal_data(_notebook("9 504 852 lignes")) == []
    assert find_personal_data(_notebook("x@users.noreply.github.com")) == []
