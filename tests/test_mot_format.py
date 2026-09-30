"""Testes da leitura e escrita do formato MOTChallenge.

Este módulo é a única fronteira do projeto onde acontecem duas conversões: quadro
1-indexado ↔ 0-indexado, e caixa ``xywh`` ↔ ``xyxy``. Erro de uma unidade aqui desalinha o
vídeo inteiro contra as anotações, e o sintoma — IDF1 baixo — é indistinguível de um
problema de associação. Daí os testes serem sobre o ida-e-volta, e não sobre o texto.

Rode com:   pytest tests/test_mot_format.py -v
"""

import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from helpers.setup_environment import setup_environment
setup_environment()

from src.data.mot_format import read_mot_gt, write_mot_gt
from src.data.sequence import Frame, Sequence
from src.data.synthetic_video import SyntheticVideos


def test_round_trip_preserva_tudo(tmp_path):
    """O quê: escrever uma sequência e ler de volta devolve as mesmas caixas, ids e
    visibilidades.
    Por quê: é o contrato que permite o sintético e o MOT17 compartilharem todo o código a
    jusante. Se a ida-e-volta perdesse informação, um bug apareceria só depois, no MOT17.
    Como: gera um vídeo, grava, relê e compara quadro a quadro.
    """
    original = SyntheticVideos(n_sequences=1, seed=1, min_frames=20, max_frames=25,
                               min_obj=4, max_obj=6)[0]

    caminho = write_mot_gt(original, tmp_path / original.name / "gt.txt")
    lida = read_mot_gt(caminho, n_frames=len(original))

    assert len(lida) == len(original)
    for a, b in zip(original, lida):
        assert a.index == b.index
        assert np.array_equal(a.ids, b.ids)
        # gravado com 2 casas decimais: a tolerância é a da própria serialização
        assert np.allclose(a.boxes, b.boxes, atol=0.01)
        assert np.allclose(a.visibility, b.visibility, atol=1e-4)


def test_quadro_e_1_indexado_em_disco(tmp_path):
    """O quê: o quadro 0 em memória vira o quadro 1 no arquivo.
    Por quê: é a convenção do MOTChallenge. Gravar 0-indexado faria as anotações ficarem um
    quadro adiantadas em relação a qualquer ferramenta externa — e em relação ao ``img1/``,
    que também numera a partir de 1.
    Como: lê a primeira coluna da primeira linha do arquivo.
    """
    seq = Sequence(name="T", fps=30.0, width=64, height=64, frames=[
        Frame(index=0, boxes=[[1, 2, 11, 22]], ids=[7], visibility=[0.5]),
    ])
    caminho = write_mot_gt(seq, tmp_path / "gt.txt")
    primeira = caminho.read_text().strip().splitlines()[0].split(",")
    assert primeira[0] == "1"
    assert primeira[1] == "7"


def test_caixa_vai_como_xywh(tmp_path):
    """O quê: a caixa ``xyxy`` (1, 2, 11, 22) é gravada como ``xywh`` (1, 2, 10, 20).
    Por quê: o ``gt.txt`` do MOT17 usa canto + tamanho. Gravar canto + canto produziria
    caixas gigantes ao ser lido por qualquer coisa que siga o formato.
    Como: confere os quatro campos de geometria da linha gravada.
    """
    seq = Sequence(name="T", fps=30.0, width=64, height=64, frames=[
        Frame(index=0, boxes=[[1, 2, 11, 22]], ids=[1], visibility=[1.0]),
    ])
    caminho = write_mot_gt(seq, tmp_path / "gt.txt")
    campos = caminho.read_text().strip().split(",")
    assert [float(c) for c in campos[2:6]] == [1.0, 2.0, 10.0, 20.0]


def test_quadros_vazios_sao_preservados(tmp_path):
    """O quê: um quadro sem nenhum objeto continua existindo na sequência lida.
    Por quê: o ``gt.txt`` não tem linha para quadro vazio. Se a leitura simplesmente pulasse
    o índice, os quadros seguintes escorregariam para trás e o vídeo sairia dessincronizado
    das anotações.
    Como: grava uma sequência com o quadro do meio vazio e confere índices e comprimento.
    """
    seq = Sequence(name="T", fps=30.0, width=64, height=64, frames=[
        Frame(index=0, boxes=[[0, 0, 5, 5]], ids=[1], visibility=[1.0]),
        Frame(index=1, boxes=np.empty((0, 4)), ids=np.empty(0), visibility=np.empty(0)),
        Frame(index=2, boxes=[[1, 1, 6, 6]], ids=[1], visibility=[1.0]),
    ])
    lida = read_mot_gt(write_mot_gt(seq, tmp_path / "gt.txt"), n_frames=3)

    assert len(lida) == 3
    assert len(lida[1]) == 0
    assert [f.index for f in lida] == [0, 1, 2]


def test_sem_coluna_de_visibilidade_assume_visivel(tmp_path):
    """O quê: uma linha no formato do ``det.txt`` (sem ``visibility``) é lida como
    totalmente visível.
    Por quê: as detecções públicas do MOT17 vêm só até ``conf``. Ler o campo ausente como
    zero marcaria toda detecção como invisível e envenenaria qualquer filtro por
    visibilidade.
    Como: escreve uma linha truncada à mão e lê.
    """
    caminho = tmp_path / "det.txt"
    caminho.write_text("1,-1,10.0,20.0,5.0,8.0,0.9\n", encoding="utf-8")
    lida = read_mot_gt(caminho)
    assert lida[0].visibility[0] == 1.0
