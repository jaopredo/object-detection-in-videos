"""Testes da Parte 2: parametrização da caixa, janelas de BPTT, modelo e rastreador.

O que se testa aqui são as propriedades que, se quebrarem, produzem um número **plausível e
errado** — que é o tipo de defeito que sobrevive até a apresentação. Invariância de escala,
o rollout realimentando a previsão certa, e o rastreador não vazando estado entre sequências.
"""

import numpy as np
import pytest
import torch

from helpers.construtores import sequencia, trilha
from metrics import evaluate
from src.boxes import (aplicar_delta, contexto, cxcywh_para_xyxy, delta,
                       xyxy_para_cxcywh)
from src.data.detections import Detections, detections_from_sequence
from src.data.windows import TrackWindows, collate, trechos_continuos
from src.models.motion_rnn import (MotionRNN, aplicar_delta_torch, delta_torch,
                                   hidden_para_orcamento)
from src.losses.motion import SmoothL1BoxLoss
from src.tracking.rnn_tracker import RNNTracker


# ================================================================ parametrização

def test_ida_e_volta_de_xyxy_para_cxcywh():
    caixas = np.array([[10., 20., 50., 100.], [0., 0., 5., 5.]])
    assert np.allclose(cxcywh_para_xyxy(xyxy_para_cxcywh(caixas)), caixas)


def test_delta_e_aplicar_delta_sao_inversos():
    a = xyxy_para_cxcywh(np.array([[10., 20., 50., 100.]]))
    b = xyxy_para_cxcywh(np.array([[18., 26., 62., 112.]]))
    assert np.allclose(aplicar_delta(a, delta(a, b)), b)


def test_o_alvo_e_invariante_a_escala():
    """O mesmo movimento relativo em duas escalas tem que dar o mesmo alvo.

    É a propriedade que permite treinar em 1920x1080 e avaliar na MOT17-05, que é 640x480.
    Sem ela a rede teria que aprender o movimento uma vez por profundidade de cena.
    """
    pequeno_de = xyxy_para_cxcywh(np.array([[0., 0., 10., 20.]]))
    pequeno_para = xyxy_para_cxcywh(np.array([[5., 0., 15., 20.]]))
    grande_de = xyxy_para_cxcywh(np.array([[0., 0., 100., 200.]]))
    grande_para = xyxy_para_cxcywh(np.array([[50., 0., 150., 200.]]))

    assert np.allclose(delta(pequeno_de, pequeno_para), delta(grande_de, grande_para))


def test_delta_de_caixa_parada_e_zero():
    a = xyxy_para_cxcywh(np.array([[10., 20., 50., 100.]]))
    assert np.allclose(delta(a, a), 0.0)


def test_mudanca_de_escala_e_simetrica_em_log():
    """Dobrar e encolher pela metade têm que ficar à mesma distância de zero."""
    base = np.array([[50., 50., 40., 40.]])
    dobro = np.array([[50., 50., 80., 80.]])
    metade = np.array([[50., 50., 20., 20.]])
    assert delta(base, dobro)[0, 2] == pytest.approx(-delta(base, metade)[0, 2])


def test_caixa_degenerada_nao_produz_nan():
    """O ruído do simulador pode gerar caixa de dimensão zero; NaN contamina o lote todo."""
    zero = np.array([[50., 50., 0., 0.]])
    normal = np.array([[50., 50., 40., 40.]])
    assert np.isfinite(delta(zero, normal)).all()
    assert np.isfinite(aplicar_delta(zero, np.array([[1., 1., 5., 5.]]))).all()


def test_versoes_numpy_e_torch_concordam():
    """São duas implementações da mesma conta; divergir daria treino e inferência diferentes."""
    a = np.array([[50., 60., 40., 80.]])
    b = np.array([[58., 66., 44., 86.]])
    assert np.allclose(
        delta(a, b),
        delta_torch(torch.tensor(a), torch.tensor(b)).numpy(),
    )
    d = np.array([[0.2, -0.1, 0.1, 0.05]])
    assert np.allclose(
        aplicar_delta(a, d),
        aplicar_delta_torch(torch.tensor(a), torch.tensor(d)).numpy(),
    )


def test_contexto_normaliza_pelo_quadro():
    caixa = np.array([[960., 540., 192., 216.]])
    c = contexto(caixa, 1920, 1080)
    assert np.allclose(c, [[0.5, 0.5, 0.1, 0.2]])


# ======================================================================= janelas

def test_trechos_continuos_quebra_nos_buracos():
    seq = sequencia({1: trilha(0, [0, 1, 2, 7, 8])}, n_frames=10)
    assert trechos_continuos(seq, 1) == [[0, 1, 2], [7, 8]]


def test_janela_nunca_atravessa_buraco_do_gabarito():
    """Uma janela que cruzasse o buraco teria alvo inventado nos quadros ausentes."""
    seq = sequencia({1: trilha(0, list(range(5)) + list(range(20, 25)))}, n_frames=25)
    janelas = TrackWindows([seq], T=4, stride=1, p_oclusao=0.0)

    for j in janelas.janelas:
        inicio = j["quadro_inicial"]
        assert inicio + 4 <= 5 or inicio >= 20, f"janela em {inicio} cruza o buraco"


def test_dois_trechos_da_mesma_identidade_nao_se_sobrescrevem():
    """A chave do índice inclui o nº do trecho; sem isso metade dos dados sumia em silêncio."""
    seq = sequencia({1: trilha(0, list(range(6)) + list(range(20, 26)))}, n_frames=26)
    assert len(TrackWindows([seq], T=4, stride=4, p_oclusao=0.0)) == 2


def test_oclusao_simulada_nunca_mascara_o_primeiro_passo():
    """Sem o passo 0 observado, o estado não tem de onde partir e a janela vira ruído."""
    seq = sequencia({i: trilha(i % 4, range(40)) for i in range(1, 9)}, n_frames=40)
    janelas = TrackWindows([seq], T=8, stride=4, p_oclusao=1.0, oclusao_max=6, seed=0)

    for i in range(len(janelas)):
        assert janelas[i]["observado"][0] == 1.0


def test_oclusao_simulada_e_contigua():
    """Oclusão real é contígua — o pedestre fica atrás do poste, não pisca."""
    seq = sequencia({i: trilha(i % 4, range(40)) for i in range(1, 9)}, n_frames=40)
    janelas = TrackWindows([seq], T=12, stride=6, p_oclusao=1.0, oclusao_max=6, seed=0)

    for i in range(len(janelas)):
        cego = (janelas[i]["observado"] == 0).nonzero().flatten().tolist()
        if cego:
            assert cego == list(range(cego[0], cego[-1] + 1)), f"buraco furado: {cego}"


def test_sem_oclusao_simulada_tudo_e_observado():
    seq = sequencia({1: trilha(0, range(20))}, n_frames=20)
    janelas = TrackWindows([seq], T=8, stride=4, p_oclusao=0.0)
    assert all(janelas[i]["observado"].min() == 1.0 for i in range(len(janelas)))


def test_aumento_de_dt_gera_mais_janelas_e_mais_valores_de_dt():
    """A correção da Parte 4: o modelo só tinha visto Δt numa faixa de 1,2x."""
    seq = sequencia({1: trilha(0, range(60))}, n_frames=60)
    sem = TrackWindows([seq], T=8, stride=4, p_oclusao=0.0)
    com = TrackWindows([seq], T=8, stride=4, p_oclusao=0.0, strides_dt=(1, 2, 3, 5))

    assert len(com) > len(sem)
    dts_sem = {float(sem[i]["dt"][0]) for i in range(len(sem))}
    dts_com = {float(com[i]["dt"][0]) for i in range(len(com))}
    assert len(dts_sem) == 1 and len(dts_com) == 4
    assert max(dts_com) == pytest.approx(5 * min(dts_com))


def test_o_lote_sabe_o_proprio_tamanho():
    """``len`` de um dicionário seria o número de chaves, e a média das épocas sairia errada."""
    seq = sequencia({i: trilha(i % 3, range(20)) for i in range(1, 6)}, n_frames=20)
    janelas = TrackWindows([seq], T=8, stride=4, p_oclusao=0.0)
    lote = collate([janelas[i] for i in range(5)])
    assert len(lote) == 5 and lote.caixas.shape == (5, 8, 4)


# ======================================================================== modelo

def test_orcamento_de_parametros_iguala_as_tres_celulas():
    """O Eixo 1 exige 'o mesmo orçamento aproximado'; a conta à mão é fácil de errar."""
    contagens = {}
    for celula in ("rnn", "lstm", "gru"):
        h = hidden_para_orcamento(celula, 20000)
        contagens[celula] = MotionRNN(celula, h).n_parametros()
        assert contagens[celula] <= 20000

    assert min(contagens.values()) / max(contagens.values()) > 0.95


def test_a_cabeca_comeca_prevendo_incremento_zero():
    """"O objeto continua onde está" é o palpite certo na média — o treino começa perto."""
    m = MotionRNN("gru", 16)
    lote = _lote(2, 5)
    saida = m.rollout(**lote)
    assert torch.allclose(saida["incremento"], torch.zeros_like(saida["incremento"]))
    assert torch.allclose(saida["previsao"][:, 0], lote["caixas"][:, 0])


def test_rollout_usa_a_observacao_onde_ha_e_a_propria_previsao_onde_nao_ha():
    """É a linha que separa treino de inferência — e as duas usam este mesmo código."""
    torch.manual_seed(0)
    m = MotionRNN("gru", 16)
    for p in m.cabeca.parameters():
        torch.nn.init.normal_(p, std=0.1)   # tira a cabeça do zero para haver o que checar

    lote = _lote(3, 8)
    lote["observado"][:, 3:6] = 0.0
    saida = m.rollout(**lote)

    assert torch.allclose(saida["crenca"][:, 2], lote["caixas"][:, 2])      # observado
    assert torch.allclose(saida["crenca"][:, 4], saida["previsao"][:, 3])   # cego
    assert not torch.allclose(saida["crenca"][:, 4], lote["caixas"][:, 4])


def test_rollout_nao_produz_nan_com_pesos_grandes():
    """O clamp do expoente existe para isto: sem ele, tw=20 vira caixa de 10^8 px."""
    torch.manual_seed(0)
    m = MotionRNN("gru", 16)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=5.0)
    saida = m.rollout(**_lote(4, 10))
    assert torch.isfinite(saida["previsao"]).all()


@pytest.mark.parametrize("celula", ["rnn", "lstm", "gru"])
def test_as_tres_celulas_rodam_e_propagam_gradiente(celula):
    m = MotionRNN(celula, 16)
    lote = _lote(3, 6)
    perda = m.rollout(**lote)["previsao"].sum()
    perda.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_reter_estados_devolve_um_por_passo():
    m = MotionRNN("lstm", 16)
    saida = m.rollout(**_lote(2, 7), reter_estados=True)
    assert len(saida["estados"]) == 7
    assert saida["estados"][0].shape == (2, 16)


def test_celula_desconhecida_levanta_erro():
    with pytest.raises(ValueError, match="cell"):
        MotionRNN("transformer")


# ========================================================================= perda

def test_perda_e_zero_quando_a_previsao_e_perfeita():
    """A perda é sobre o resíduo: zero quando prevista e verdadeira coincidem."""
    lote = collate([_amostra(6)])
    saida = {"previsao": lote.caixas[:, 1:].clone()}
    saida["previsao"] = torch.cat([saida["previsao"], lote.caixas[:, -1:]], dim=1)
    perda, _ = SmoothL1BoxLoss()(saida, lote)
    assert float(perda) == pytest.approx(0.0, abs=1e-9)


def test_peso_sob_oclusao_concentra_a_perda_nos_passos_cegos():
    """Errar mais nos passos cegos tem que doer mais quando o peso está ligado.

    O erro precisa ser **diferente** entre passos observados e cegos para o teste medir
    algo: com o mesmo erro em todo passo, qualquer ponderação dá a mesma média ponderada —
    foi assim que a primeira versão deste teste passou a comparar 0,0264 com 0,0264.
    """
    lote = collate([_amostra(8)])
    lote.observado[:, 3:6] = 0.0

    previsao = lote.caixas.clone()
    previsao[:, :, 0] += 2.0          # erro pequeno em todo passo
    previsao[:, 2:5, 0] += 30.0       # erro grande justamente nos passos que viram cegos

    sem, comp_sem = SmoothL1BoxLoss(peso_sob_oclusao=1.0)({"previsao": previsao}, lote)
    com, comp_com = SmoothL1BoxLoss(peso_sob_oclusao=5.0)({"previsao": previsao}, lote)

    # as componentes confirmam que o erro está mesmo concentrado nos passos cegos
    assert comp_sem["cego"] > comp_sem["visto"]
    # e o peso faz a perda total subir na direção deles
    assert float(com) > float(sem)


# ==================================================================== rastreador

def test_rastreador_recorrente_com_deteccao_perfeita_acerta_tudo():
    """Piso: sem ruído e com detecção em todo quadro, o rastreador não pode errar."""
    gabarito = sequencia({1: trilha(0, range(12)), 2: trilha(2, range(12))}, n_frames=12)
    m = MotionRNN("gru", 16)
    pred = RNNTracker(m, iou_threshold=0.3, max_age=10, min_hits=1).run(
        detections_from_sequence(gabarito), "T", 30.0, 1000, 1000)
    assert evaluate(gabarito, pred).idf1 == 1.0


def test_rastreador_recorrente_nao_vaza_estado_entre_sequencias():
    a = sequencia({1: trilha(0, range(8))}, n_frames=8)
    b = sequencia({1: trilha(0, range(8))}, n_frames=8)
    tracker = RNNTracker(MotionRNN("gru", 16), min_hits=1)

    tracker.run(detections_from_sequence(a), "A", 30.0, 1000, 1000)
    pred = tracker.run(detections_from_sequence(b), "B", 30.0, 1000, 1000)
    assert evaluate(b, pred).idf1 == 1.0


@pytest.mark.parametrize("celula", ["rnn", "lstm", "gru"])
def test_o_estado_em_lote_volta_para_a_track_certa(celula):
    """O estado é empilhado por track e refatiado; trocar as linhas daria tudo errado.

    Duas tracks com movimentos **diferentes** e um modelo com pesos não triviais: se o
    estado fosse devolvido ao dono errado, a previsão de cada uma sairia com a dinâmica da
    outra e o rastreamento quebraria.
    """
    torch.manual_seed(0)
    m = MotionRNN(celula, 16)
    for p in m.cabeca.parameters():
        torch.nn.init.normal_(p, std=0.05)

    parado = {t: [0, 0, 40, 40] for t in range(10)}
    andando = {t: [300 + 8 * t, 200, 340 + 8 * t, 240] for t in range(10)}
    gabarito = sequencia({1: parado, 2: andando}, n_frames=10)

    pred = RNNTracker(m, iou_threshold=0.2, max_age=10, min_hits=1).run(
        detections_from_sequence(gabarito), "T", 30.0, 1000, 1000)
    assert evaluate(gabarito, pred).n_pred_ids == 2


def test_sequencia_sem_deteccao_nao_quebra_o_rastreador_recorrente():
    gabarito = sequencia({1: trilha(0, range(5))}, n_frames=5)
    vazias = [Detections(t, np.empty((0, 4)), np.empty(0)) for t in range(5)]
    pred = RNNTracker(MotionRNN("gru", 16), min_hits=1).run(vazias, "T", 30.0, 1000, 1000)
    assert len(pred) == 5 and len(pred.track_ids) == 0


# ------------------------------------------------------------------------ apoio

def _amostra(T: int) -> dict:
    caixas = torch.stack([
        torch.tensor([100.0 + 5 * t, 200.0, 40.0, 80.0]) for t in range(T)
    ])
    return {"caixas": caixas, "observado": torch.ones(T),
            "dt": torch.full((T,), 1 / 30), "score": torch.ones(T),
            "tamanho": torch.tensor([1920.0, 1080.0])}


def _lote(B: int, T: int) -> dict:
    amostras = [_amostra(T) for _ in range(B)]
    junta = lambda k: torch.stack([a[k] for a in amostras])
    return {"caixas": junta("caixas"), "observado": junta("observado"),
            "dt": junta("dt"), "score": junta("score"), "tamanho": junta("tamanho")}
