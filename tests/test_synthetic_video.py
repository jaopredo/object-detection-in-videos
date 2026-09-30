"""Testes do gerador de vídeos sintéticos — artefato 1 da Parte 0.

O enunciado chama isso de *requisito verificável*: "as elipses precisam ser desenhadas com
ordem de profundidade, de modo que uma passe atrás da outra e realmente desapareça, se não
não é oclusão". Um gerador que só desenha por cima sem medir nada passaria despercebido até
a Parte 4, quando o horizonte de memória fosse comparado com uma distribuição de oclusão
que nunca existiu.

Cada teste é documentado com:
  - **O quê**: o que o teste verifica
  - **Por quê**: por que o teste é necessário
  - **Como**: mecânica do teste

Rode com:   pytest tests/test_synthetic_video.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from helpers.setup_environment import setup_environment
setup_environment()

from src.data.synthetic_video import MovingEllipse, SyntheticVideos
from src.evaluation.occlusion import longest_run, occlusion_runs


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def ds():
    """Conjunto pequeno, com a oclusão calibrada em 10 quadros."""
    return SyntheticVideos(
        n_sequences=3, size=128, seed=42,
        min_frames=50, max_frames=60, min_obj=5, max_obj=8,
        speed=2.0, occlusion_duration=10,
    )


# --------------------------------------------------------------------------- formato

def test_formato_dos_quadros(ds):
    """O quê: cada quadro traz boxes (N,4), ids inteiros e visibility em [0,1], com o
    mesmo N nos três.
    Por quê: todo o resto do projeto — métricas, tracker, figuras — assume esse contrato.
    Um N desalinhado entre caixas e ids associaria a caixa de um objeto à identidade de
    outro, e o IDF1 resultante seria silenciosamente errado.
    Como: percorre uma sequência inteira conferindo formas e faixas.
    """
    seq = ds[0]
    for frame in seq:
        n = len(frame.ids)
        assert frame.boxes.shape == (n, 4)
        assert frame.visibility.shape == (n,)
        assert frame.ids.dtype == np.int64
        assert np.all(frame.visibility >= 0.0) and np.all(frame.visibility <= 1.0)
        # caixa bem formada: canto superior esquerdo antes do inferior direito
        assert np.all(frame.boxes[:, 2] > frame.boxes[:, 0])
        assert np.all(frame.boxes[:, 3] > frame.boxes[:, 1])


def test_comprimento_na_faixa_pedida(ds):
    """O quê: o número de quadros de cada vídeo respeita [min_frames, max_frames].
    Por quê: o enunciado pede vídeos de 30 a 60 quadros; se o gerador ignorasse a faixa,
    a janela de BPTT da Parte 2 seria dimensionada contra um número errado.
    Como: confere o comprimento das três sequências.
    """
    for i in range(len(ds)):
        assert 50 <= len(ds[i]) <= 60


def test_identidade_unica_por_quadro(ds):
    """O quê: nenhuma identidade aparece duas vezes no mesmo quadro.
    Por quê: é a definição de identidade, e a atribuição um-para-um do IDF1 depende disso.
    Um id duplicado quebraria a premissa do matching húngaro sem levantar erro.
    Como: compara o número de ids com o número de ids distintos. (O ``Frame`` já valida
    isso na construção; o teste garante que a validação não seja removida por engano.)
    """
    for i in range(len(ds)):
        for frame in ds[i]:
            assert len(np.unique(frame.ids)) == len(frame.ids)


# --------------------------------------------------------------------- reprodutibilidade

def test_mesma_seed_mesmos_videos():
    """O quê: duas instâncias com a mesma seed produzem quadros e caixas idênticos.
    Por quê: a ablação da Parte 3 mede média ± desvio sobre 3 seeds. Se os dados variassem
    entre execuções da mesma seed, o desvio medido seria ruído do gerador em vez do efeito
    do experimento — exatamente o que a ablação deveria isolar.
    Como: gera duas vezes e compara pixels e anotações.
    """
    a = SyntheticVideos(n_sequences=2, seed=7, min_frames=30, max_frames=35, max_obj=6)
    b = SyntheticVideos(n_sequences=2, seed=7, min_frames=30, max_frames=35, max_obj=6)

    for i in range(2):
        assert len(a[i]) == len(b[i])
        for fa, fb in zip(a[i], b[i]):
            assert np.array_equal(fa.image, fb.image)
            assert np.array_equal(fa.boxes, fb.boxes)
            assert np.array_equal(fa.ids, fb.ids)


def test_seeds_diferentes_videos_diferentes():
    """O quê: seeds diferentes produzem vídeos diferentes.
    Por quê: os três splits são gerados pelo mesmo código com seeds deslocadas. Se a seed
    não mudasse o resultado, treino, validação e teste seriam o mesmo vídeo e toda a
    separação seria de fachada.
    Como: compara o primeiro quadro de duas seeds.
    """
    a = SyntheticVideos(n_sequences=1, seed=0, min_frames=30, max_frames=31)
    b = SyntheticVideos(n_sequences=1, seed=999, min_frames=30, max_frames=31)
    assert not np.array_equal(a[0][0].image, b[0][0].image)


# --------------------------------------------------------------------------- oclusão

def test_oclusao_e_real_e_a_duracao_e_a_pedida(ds):
    """O quê: o objeto designado (id 1) fica com visibilidade exatamente 0 por um trecho
    contínuo cujo comprimento é o ``occlusion_duration`` pedido, dentro de ±1 quadro.
    Por quê: é o *requisito verificável* do enunciado. Sem ele, "oclusão" seria só uma
    elipse desenhada por cima de outra, e a duração — que a Parte 4 usa como régua do
    horizonte de memória — não significaria nada.
    Como: para cada sequência, pega o maior trecho de visibilidade zero do id 1 e compara
    com o valor pedido no construtor.
    """
    for i in range(len(ds)):
        seq = ds[i]
        runs = [r for r in occlusion_runs(seq) if r.track_id == 1]
        assert runs, f"{seq.name}: o objeto designado nunca ficou totalmente escondido"

        duracao = max(r.duration for r in runs)
        assert abs(duracao - 10) <= 1, (
            f"{seq.name}: pedido 10 quadros de oclusão, obtido {duracao}"
        )


def test_identidade_sobrevive_a_oclusao(ds):
    """O quê: depois de sumir, o objeto volta com o MESMO id.
    Por quê: é o problema inteiro do assignment. Se o gerador emitisse um id novo na volta,
    o ground truth já conteria o ID switch que o modelo deveria evitar, e o IDF1 teria um
    teto artificial que ninguém conseguiria explicar.
    Como: confirma que o id reaparece depois do vão e que a caixa existe dos dois lados.
    """
    seq = ds[0]
    run = longest_run(seq)
    assert run is not None

    assert seq[run.last_seen].box_of(run.track_id) is not None
    assert seq[run.next_seen].box_of(run.track_id) is not None
    # o id continua anotado DURANTE a oclusão, com visibilidade zero — é a caixa amodal
    assert seq[run.start].box_of(run.track_id) is not None
    assert seq.visibility_of(run.track_id)[run.start] == 0.0


def test_z_order_quem_esta_atras_e_que_some():
    """O quê: com duas elipses sobrepostas, a de z menor perde visibilidade e a de z maior
    permanece inteira.
    Por quê: se a ordem de desenho e o cálculo de visibilidade discordassem, a imagem
    mostraria uma coisa e a anotação afirmaria outra — e o modelo seria treinado contra um
    ground truth que não corresponde ao pixel.
    Como: monta duas elipses no mesmo centro, renderiza um quadro e lê as visibilidades.
    """
    ds = SyntheticVideos(n_sequences=1, seed=0, min_frames=1, max_frames=2, min_obj=2, max_obj=2)
    tamanho = ds.size

    atras = MovingEllipse(id=1, center=np.array([64.0, 64.0]), velocity=np.zeros(2),
                          r_radius=10, c_radius=10, rotation=0.0,
                          color=np.array([1.0, 0.0, 0.0]), z=0)
    frente = MovingEllipse(id=2, center=np.array([64.0, 64.0]), velocity=np.zeros(2),
                           r_radius=14, c_radius=14, rotation=0.0,
                           color=np.array([0.0, 1.0, 0.0]), z=1)

    rng = np.random.RandomState(0)
    # sem occluders e sem ruído: o teste isola a ordem de profundidade
    quadro = ds._render([atras, frente], [], 0, rng, bg=0.0, gain=1.0, offset=0.0)

    vis = dict(zip(quadro.ids.tolist(), quadro.visibility.tolist()))
    assert vis[1] == pytest.approx(0.0, abs=1e-6), "a elipse de trás deveria sumir"
    assert vis[2] == pytest.approx(1.0, abs=1e-6), "a elipse da frente deveria ficar inteira"


def test_caixa_e_amodal_durante_a_oclusao(ds):
    """O quê: a caixa do objeto escondido tem o mesmo tamanho de quando ele está visível.
    Por quê: é a convenção do MOT17 e é o que dá sentido ao campo ``visibility`` — a caixa
    diz onde o objeto está, a visibilidade diz quanto dele se vê. Uma caixa que encolhesse
    com a oclusão faria o alvo de regressão da trilha A ensinar o modelo a encolher junto.
    Como: compara a área da caixa no último quadro visível e no meio da oclusão.
    """
    seq = ds[0]
    run = longest_run(seq)

    def area(t):
        x1, y1, x2, y2 = seq[t].box_of(run.track_id)
        return (x2 - x1) * (y2 - y1)

    meio = (run.start + run.end) // 2
    # tolerância de 20%: a elipse gira enquanto atravessa, então a caixa do retângulo
    # envolvente muda um pouco. O que não pode é colapsar.
    assert area(meio) == pytest.approx(area(run.last_seen), rel=0.2)


def test_occlusion_duration_maior_gera_sumico_maior():
    """O quê: dobrar ``occlusion_duration`` dobra o tempo que o objeto fica escondido.
    Por quê: é o botão que a Parte 1 vai girar para mostrar onde o baseline quebra. Se ele
    não tivesse efeito monotônico, a curva de degradação não mediria nada.
    Como: gera com 5 e com 20 e compara o maior trecho do objeto designado.
    """
    def sumico(duracao):
        ds = SyntheticVideos(n_sequences=1, seed=3, min_frames=70, max_frames=80,
                             min_obj=5, max_obj=6, speed=2.0, occlusion_duration=duracao)
        runs = [r for r in occlusion_runs(ds[0]) if r.track_id == 1]
        return max(r.duration for r in runs)

    assert sumico(5) < sumico(20)
