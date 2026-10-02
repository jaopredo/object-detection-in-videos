"""Testes do grid search de hiperparâmetros (Parte 2b) que não exigem o MOT17 baixado.

O que depende de dado real em disco (treinar de verdade, avaliar IDF1) fica fora daqui — é
coberto pelo smoke test manual descrito no plano. O que entra: o holdout interno não vaza
nem se sobrepõe ao treino oficial, e os overrides dotted-path do YAML aplicam na chave certa.

Rode com:   pytest tests/test_grid_search.py -v
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))

from src.data.mot17 import SPLITS, sequencias_do_split
from src.gridsearch.runner import GridSearchRunner, _set_dotted


def test_grid_val_e_so_mot17_02():
    """O quê: o holdout interno do grid search é exatamente a MOT17-02.
    Por quê: é a decisão tomada com o usuário — a menor das 4 sequências de treino, para não
    distorcer demais a proporção treino/validação.
    """
    assert SPLITS["grid_val"] == ("MOT17-02",)


def test_grid_train_mais_grid_val_fecha_o_train_oficial():
    """O quê: `grid_train` ∪ `grid_val` é exatamente `SPLITS["train"]`, sem sobra e sem
    sobreposição.
    Por quê: o holdout é um corte **dentro** do treino oficial — se sobrasse ou faltasse
    sequência, o grid search estaria vendo dado de treino que o `train` normal não vê (ou
    vice-versa), e a busca deixaria de ser comparável com o treino de produção.
    """
    grid_train = set(sequencias_do_split("grid_train"))
    grid_val = set(sequencias_do_split("grid_val"))

    assert grid_train.isdisjoint(grid_val)
    assert grid_train | grid_val == set(SPLITS["train"])


def test_grid_val_nao_e_o_val_oficial():
    """O quê: `grid_val` (MOT17-02) é disjunto do `val` oficial do projeto (MOT17-10).
    Por quê: são dois conceitos diferentes — `val` decide época/checkpoint e tuning de
    tracking; `grid_val` decide hiperparâmetro. Se colidissem, uma mudança num afetaria o
    outro silenciosamente.
    """
    assert set(SPLITS["grid_val"]).isdisjoint(SPLITS["val"])


def test_set_dotted_cria_chaves_aninhadas():
    raw = {"train": {"lr": 0.001}}
    _set_dotted(raw, "train.lr", 0.003)
    _set_dotted(raw, "model.param_budget", 40000)

    assert raw["train"]["lr"] == 0.003
    assert raw["model"]["param_budget"] == 40000


def test_grid_search_runner_le_params_e_seeds(tmp_path):
    """O quê: `GridSearchRunner` lê `grid_search.params`/`seeds` do YAML, sem precisar de
    nenhum hiperparâmetro específico hardcoded em código.
    Por quê: é o pedido explícito — adicionar uma chave em `params` deve bastar para entrar
    na grade.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 0\n"
        "data: {kind: mot17, root: x}\n"
        "output_dir: outputs/x\n"
        "grid_search:\n"
        "  seeds: [0, 1]\n"
        "  params:\n"
        "    train.lr: [0.001, 0.003]\n"
        "    model.param_budget: [10000]\n",
        encoding="utf-8",
    )
    runner = GridSearchRunner(str(config))

    assert runner.seeds == (0, 1)
    assert runner.params == {"train.lr": [0.001, 0.003], "model.param_budget": [10000]}


def test_grid_search_runner_exige_bloco_grid_search(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 0\ndata: {kind: mot17, root: x}\noutput_dir: outputs/x\n", encoding="utf-8",
    )
    with pytest.raises(ValueError):
        GridSearchRunner(str(config))
