"""Testes do simulador de detector — artefato 2 da Parte 0.

O simulador é usado em três lugares (validar a Parte 1 no sintético, o experimento de
qualidade de detecção, e construir casos de teste da métrica), então um defeito aqui
apareceria como resultado errado em três partes diferentes do trabalho — e pareceria um
problema do rastreador nas três.
"""

import numpy as np
import pytest

from helpers.construtores import sequencia, trilha
from src.data.detections import Detections, detections_from_sequence, read_mot_det, write_mot_det
from src.data.detector_sim import simulate_detections


@pytest.fixture
def seq():
    return sequencia({1: trilha(0, range(10)),
                      2: trilha(1, range(10)),
                      3: trilha(2, range(10))}, n_frames=10)


def test_sem_defeito_nenhum_devolve_o_gabarito_bit_a_bit(seq):
    """O piso do simulador: zero em todos os botões tem que ser a identidade."""
    dets = simulate_detections(seq, drop_p=0.0, coord_noise=0.0, fp_rate=0.0, rng=0)

    assert len(dets) == len(seq)
    for d, frame in zip(dets, seq):
        assert np.array_equal(d.boxes, frame.boxes)
        assert np.all(d.scores == 1.0)


def test_a_mesma_seed_da_exatamente_as_mesmas_deteccoes_estragadas(seq):
    """Sem isto não dá para comparar duas regras de associação sobre a mesma entrada ruim."""
    kw = dict(drop_p=0.3, coord_noise=0.05, fp_rate=2.0)
    a = simulate_detections(seq, **kw, rng=7)
    b = simulate_detections(seq, **kw, rng=7)
    c = simulate_detections(seq, **kw, rng=8)

    assert all(np.array_equal(x.boxes, y.boxes) for x, y in zip(a, b))
    assert not all(np.array_equal(x.boxes, y.boxes) for x, y in zip(a, c))


def test_descartar_tudo_nao_deixa_nenhuma_caixa_verdadeira(seq):
    dets = simulate_detections(seq, drop_p=1.0, rng=0)
    assert sum(len(d) for d in dets) == 0


def test_drop_p_descarta_aproximadamente_a_fracao_pedida():
    """30 objetos x 40 quadros = 1200 sorteios: a média tem que bater com o pedido."""
    grande = sequencia({i: trilha(i % 6, range(40)) for i in range(1, 31)}, n_frames=40)
    total = sum(len(f) for f in grande)

    for p in (0.1, 0.5, 0.9):
        dets = simulate_detections(grande, drop_p=p, rng=0)
        observado = 1 - sum(len(d) for d in dets) / total
        assert observado == pytest.approx(p, abs=0.03), f"drop_p={p}"


def test_falsos_positivos_aparecem_na_taxa_pedida(seq):
    """``fp_rate`` é o número esperado por quadro, não uma fração dos objetos."""
    verdadeiras = sum(len(f) for f in seq)
    for taxa in (1.0, 5.0):
        dets = simulate_detections(seq, fp_rate=taxa, rng=0)
        extras = (sum(len(d) for d in dets) - verdadeiras) / len(seq)
        assert extras == pytest.approx(taxa, abs=0.8), f"fp_rate={taxa}"


def test_falso_positivo_tem_o_tamanho_dos_objetos_reais(seq):
    """Lixo com tamanho sorteado no quadro seria filtrável por área — e o teste seria fácil."""
    dets = simulate_detections(seq, drop_p=1.0, fp_rate=8.0, rng=0)   # só lixo
    larguras = np.concatenate([d.boxes[:, 2] - d.boxes[:, 0] for d in dets if len(d)])
    assert np.allclose(larguras, 40.0)          # o lado usado por `construtores.caixa`


def test_ruido_cresce_com_o_botao_e_e_proporcional_ao_tamanho_da_caixa():
    """σ é fração da diagonal: a mesma fração tem que deslocar mais a caixa maior."""
    pequena = sequencia({1: {0: [0, 0, 10, 10]}}, n_frames=1)
    grande = sequencia({1: {0: [0, 0, 200, 200]}}, n_frames=1)

    def desvio(seq, ruido):
        amostras = [
            simulate_detections(seq, coord_noise=ruido, rng=s)[0].boxes[0]
            for s in range(400)
        ]
        return np.std(np.array(amostras) - seq[0].boxes[0])

    assert desvio(pequena, 0.1) > desvio(pequena, 0.01)
    assert desvio(grande, 0.05) > 5 * desvio(pequena, 0.05)


def test_ruido_nunca_produz_caixa_com_cantos_invertidos():
    """x2 < x1 daria IoU negativa com tudo — um bug que se propaga até a métrica."""
    minusculas = sequencia({i: {0: [10 * i, 0, 10 * i + 4, 4]} for i in range(1, 20)}, n_frames=1)
    dets = simulate_detections(minusculas, coord_noise=2.0, rng=0)   # ruído absurdo

    for d in dets:
        assert np.all(d.boxes[:, 2] >= d.boxes[:, 0])
        assert np.all(d.boxes[:, 3] >= d.boxes[:, 1])


def test_saida_vem_ordenada_por_score_decrescente(seq):
    """É a ordem que o NMS entrega — e a que o enunciado aponta como enganosa."""
    for d in simulate_detections(seq, drop_p=0.2, fp_rate=3.0, rng=0):
        assert np.all(np.diff(d.scores) <= 0)


def test_objeto_totalmente_escondido_nao_e_detectado():
    """A caixa do gabarito é amodal; o detector só pode ver o que tem pixel na tela.

    Sem este descarte o oráculo enxerga através das coisas, e girar o botão de duração da
    oclusão no gerador não muda nada — foi o que aconteceu na primeira medição da Parte 0
    (IDF1 = 1,0000 para oclusões de 0, 5, 10 e 20 quadros).
    """
    from src.data.sequence import Frame, Sequence
    quadro = Frame(
        index=0,
        boxes=np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=np.float32),
        ids=np.array([1, 2, 3]),
        visibility=np.array([1.0, 0.5, 0.0], dtype=np.float32),
    )
    seq = Sequence(name="T", fps=30.0, width=100, height=100, frames=[quadro])

    assert len(detections_from_sequence(seq)[0]) == 3                       # sem filtro
    assert len(detections_from_sequence(seq, min_visibility=0.0)[0]) == 2   # cai a invisível
    assert len(detections_from_sequence(seq, min_visibility=0.4)[0]) == 2
    assert len(detections_from_sequence(seq, min_visibility=0.6)[0]) == 1
    assert len(simulate_detections(seq, min_visibility=0.0, rng=0)[0]) == 2


def test_botoes_invalidos_levantam_erro(seq):
    with pytest.raises(ValueError):
        simulate_detections(seq, drop_p=1.5)
    with pytest.raises(ValueError):
        simulate_detections(seq, fp_rate=-1.0)


# ------------------------------------------------------------------ ida e volta em disco

def test_det_txt_sobrevive_ao_ida_e_volta(tmp_path, seq):
    original = simulate_detections(seq, drop_p=0.2, coord_noise=0.02, fp_rate=2.0, rng=3)
    write_mot_det(original, tmp_path / "det.txt")
    lido = read_mot_det(tmp_path / "det.txt", n_frames=len(seq))

    assert len(lido) == len(original)
    for a, b in zip(original, lido):
        assert a.frame_index == b.frame_index
        assert np.allclose(a.boxes, b.boxes, atol=0.01)   # o arquivo guarda 2 casas
        assert np.allclose(a.scores, b.scores, atol=0.0001)


def test_det_txt_preserva_quadros_sem_deteccao(tmp_path):
    dets = [Detections(0, [[0, 0, 10, 10]], [0.9]),
            Detections(1, np.empty((0, 4)), np.empty(0)),
            Detections(2, [[5, 5, 15, 15]], [0.8])]
    write_mot_det(dets, tmp_path / "det.txt")
    lido = read_mot_det(tmp_path / "det.txt", n_frames=3)

    assert [len(d) for d in lido] == [1, 0, 1]
    assert [d.frame_index for d in lido] == [0, 1, 2]


def test_quadro_do_det_txt_e_1_indexado_em_disco(tmp_path):
    write_mot_det([Detections(0, [[1, 2, 3, 4]], [1.0])], tmp_path / "det.txt")
    assert (tmp_path / "det.txt").read_text().startswith("1,-1,")
