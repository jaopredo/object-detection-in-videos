"""Testes do NMS próprio e do rastreador ingênuo da Parte 1.

Sobre não usar ``torchvision.ops.nms`` como referência: o enunciado o proíbe em letras
claras. Usá-lo "só no teste" para conferir seria contornar a proibição pela porta dos
fundos — e, pior, esconderia o fato de que a nossa implementação nunca foi pensada de
verdade. Os casos abaixo são construídos à mão, com a resposta certa calculada no próprio
teste, como nos testes da métrica.
"""

import numpy as np
import pytest

from helpers.construtores import caixa, sequencia, trilha
from metrics import evaluate
from src.data.detections import Detections, detections_from_sequence
from src.data.detector_sim import simulate_detections
from src.data.synthetic_video import SyntheticVideos
from src.tracking import IoUTracker, nms, nms_por_classe


# ============================================================================== NMS

def test_nms_suprime_a_duplicata_e_mantem_a_de_maior_score():
    boxes = np.array([[0, 0, 10, 10],      # score 0,9 — sobrevive
                      [1, 1, 11, 11],      # quase a mesma, score 0,8 — suprimida
                      [50, 50, 60, 60]])   # longe, score 0,7 — sobrevive
    scores = np.array([0.9, 0.8, 0.7])

    assert list(nms(boxes, scores, 0.5)) == [0, 2]


def test_nms_devolve_em_ordem_de_score_decrescente():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]])
    scores = np.array([0.2, 0.9, 0.5])
    assert list(nms(boxes, scores, 0.5)) == [1, 2, 0]


def test_limiar_do_nms_decide_o_que_sobrevive():
    """Duas caixas com IoU = 50/150 ≈ 0,333: o limiar em volta disso vira a resposta."""
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
    scores = np.array([0.9, 0.8])

    assert len(nms(boxes, scores, 0.30)) == 1   # 0,333 > 0,30 → suprime
    assert len(nms(boxes, scores, 0.40)) == 2   # 0,333 < 0,40 → mantém


def test_nms_nao_suprime_em_cadeia():
    """A suprime B, mas C não encosta em A: C tem que sobreviver.

    As três caixas têm 10 de lado e passo 6, então IoU(A,B) = IoU(B,C) = 4·10 / 160 = 0,25
    e IoU(A,C) = 0 (elas nem se tocam). Com limiar 0,2, A suprime B; C é comparada **contra
    A**, não contra B, e sobrevive.

    O erro clássico é remover C junto, por transitividade a partir de B. O NMS compara
    sempre contra a caixa escolhida, nunca contra uma já suprimida.
    """
    boxes = np.array([[0, 0, 10, 10], [6, 0, 16, 10], [12, 0, 22, 10]])
    scores = np.array([0.9, 0.8, 0.7])

    assert list(nms(boxes, scores, 0.2)) == [0, 2]
    # no limiar exato de 0,25 a supressão é "acima de", então ninguém cai
    assert list(nms(boxes, scores, 0.25)) == [0, 1, 2]


def test_nms_com_entrada_vazia_devolve_vazio_e_nao_erro():
    assert len(nms(np.empty((0, 4)), np.empty(0), 0.5)) == 0


def test_nms_de_uma_caixa_so_devolve_ela():
    assert list(nms(np.array([[0, 0, 10, 10]]), np.array([0.5]), 0.5)) == [0]


def test_nms_e_estavel_no_empate_de_score():
    """Empate resolvido pelo índice: o resultado não pode depender do algoritmo de sort."""
    boxes = np.array([[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]])
    scores = np.array([0.5, 0.5, 0.5])
    assert list(nms(boxes, scores, 0.5)) == [0, 1, 2]


def test_nms_recusa_tamanhos_incompativeis():
    with pytest.raises(ValueError):
        nms(np.zeros((3, 4)), np.zeros(2), 0.5)


def test_nms_por_classe_nao_suprime_entre_classes_diferentes():
    """Pedestre em cima de bicicleta não é detecção duplicada."""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    scores = np.array([0.9, 0.8])

    assert len(nms(boxes, scores, 0.5)) == 1
    assert len(nms_por_classe(boxes, scores, np.array([1, 2]), 0.5)) == 2
    assert len(nms_por_classe(boxes, scores, np.array([1, 1]), 0.5)) == 1


# ======================================================================== rastreador

def _rastreia(gabarito, dets=None, **kw):
    dets = dets if dets is not None else detections_from_sequence(gabarito)
    tracker = IoUTracker(**kw)
    predicao = tracker.run(dets, gabarito.name, gabarito.fps, gabarito.width, gabarito.height)
    return predicao, evaluate(gabarito, predicao)


def test_deteccao_perfeita_e_objetos_parados_da_idf1_um():
    """O piso do rastreador: sem ruído e sem movimento, não há como errar."""
    gabarito = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
    _, m = _rastreia(gabarito, min_hits=1)

    assert m.idf1 == 1.0
    assert m.id_switches == 0 and m.count_error == 0


def test_objeto_que_some_alem_do_max_age_perde_a_identidade():
    """``max_age`` é o teto duro do horizonte de memória — a Parte 4 mede contra isto."""
    gabarito = sequencia({1: trilha(0, list(range(5)) + list(range(15, 20)))}, n_frames=20)

    # buraco de 10 quadros: com max_age=3 a track morre e volta com id novo
    _, curto = _rastreia(gabarito, min_hits=1, max_age=3)
    assert curto.n_pred_ids == 2 and curto.id_switches == 1

    # com folga suficiente, a mesma identidade atravessa
    _, longo = _rastreia(gabarito, min_hits=1, max_age=15)
    assert longo.n_pred_ids == 1 and longo.id_switches == 0
    assert longo.idf1 == 1.0


def test_min_hits_descarta_o_falso_positivo_isolado():
    """Uma detecção que aparece num quadro só não pode virar uma identidade."""
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    dets = detections_from_sequence(gabarito)
    dets[4] = Detections(4, np.vstack([dets[4].boxes, [[500, 500, 540, 540]]]),
                         np.append(dets[4].scores, 1.0))

    _, sem_filtro = _rastreia(gabarito, dets=dets, min_hits=1)
    _, com_filtro = _rastreia(gabarito, dets=dets, min_hits=3)

    assert sem_filtro.n_pred_ids == 2      # o lixo virou identidade
    assert com_filtro.n_pred_ids == 1      # filtrado
    assert com_filtro.idf1 == 1.0


def test_min_hits_emite_retroativamente_e_nao_perde_o_comeco_da_track():
    """Confirmar em 3 observações não pode custar os 2 primeiros quadros de cada track.

    A avaliação é offline: quando a track é confirmada, as caixas guardadas são despejadas
    nos quadros de onde vieram. Jogá-las fora seria perder ``min_hits − 1`` quadros de toda
    identidade, de graça.
    """
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    predicao, m = _rastreia(gabarito, min_hits=3)

    assert len(predicao[0]) == 1 and len(predicao[1]) == 1   # os dois primeiros lá
    assert m.idf1 == 1.0


def test_objetos_rapidos_perdem_a_identidade_e_e_por_isso_que_a_parte_2_existe():
    """Sem modelo de movimento, IoU entre quadros vizinhos some quando o objeto anda.

    O objeto se desloca 50 px por quadro e tem 40 px de lado: a caixa de t não encosta na de
    t−1, a IoU é 0, e cada quadro cria uma identidade nova. É o fracasso que o enunciado
    manda quantificar na Parte 1.
    """
    rapido = sequencia({1: {t: [50 * t, 0, 50 * t + 40, 40] for t in range(8)}}, n_frames=8)
    _, m = _rastreia(rapido, min_hits=1)

    assert m.n_pred_ids == 8
    assert m.idf1 < 0.3


def test_extrapolacao_nao_faz_bootstrap_sozinha():
    """Velocidade não salva o objeto rápido — e a razão importa para a Parte 2.

    A extrapolação precisa de **duas** observações para estimar o deslocamento. Se a
    primeira associação já falha (IoU zero entre quadros vizinhos), a segunda observação
    nunca chega à mesma track, e o modelo de movimento nunca liga. Ele só ajuda quem já
    estava sendo seguido.

    É a diferença de natureza que a Parte 2 explora: o estado recorrente também precisa
    ser inicializado, mas ele aprende uma dinâmica **típica** do dataset, e não uma
    diferença finita entre duas caixas.
    """
    rapido = sequencia({1: {t: [50 * t, 0, 50 * t + 40, 40] for t in range(8)}}, n_frames=8)

    _, sem = _rastreia(rapido, min_hits=1, velocidade=False)
    _, com = _rastreia(rapido, min_hits=1, velocidade=True)

    assert sem.n_pred_ids == com.n_pred_ids == 8


def test_velocidade_constante_atravessa_a_oclusao_que_a_posicao_constante_perde():
    """Onde a extrapolação de fato ganha: o buraco.

    Objeto de 40 px andando 25 px por quadro, sem detecção nos quadros 4 a 7. Quando ele
    reaparece no quadro 8 já andou 125 px desde a última observação — a caixa parada não
    encosta nele nem de longe. A extrapolação prevê ``última + velocidade·(age+1)``, que
    aponta exatamente para onde ele está.

    É o baseline honesto da Parte 2: ganhar da posição constante é fácil e não prova nada
    sobre recorrência; ganhar **disto** é que é o teste.
    """
    caixas = {t: [25 * t, 0, 25 * t + 40, 40] for t in range(12)}
    gabarito = sequencia({1: caixas}, n_frames=12)

    dets = detections_from_sequence(gabarito)
    for t in (4, 5, 6, 7):
        dets[t] = Detections(t, np.empty((0, 4)), np.empty(0))

    _, parado = _rastreia(gabarito, dets=dets, min_hits=1, max_age=10, iou_threshold=0.2)
    _, andando = _rastreia(gabarito, dets=dets, min_hits=1, max_age=10, iou_threshold=0.2,
                           velocidade=True)

    assert parado.n_pred_ids == 2 and parado.id_switches == 1
    assert andando.n_pred_ids == 1 and andando.id_switches == 0
    assert andando.idf1 > parado.idf1


def test_guloso_e_hungarian_dao_numeros_diferentes_no_mesmo_video():
    """O enunciado avisa que regras diferentes dão números diferentes. Aqui está o caso.

    Elas só discordam quando há disputa de verdade — cena densa, objetos rápidos, limiar
    baixo. Medido em 30/09: com 10 objetos a 3 px/quadro e limiar 0,3 as duas dão o **mesmo**
    IDF1 até a última casa; com 15 objetos a 5 px/quadro e limiar 0,2 elas divergem.

    Achado que vale para a apresentação: nas configurações mais duras o **guloso ganha**
    (IDF1 0,2217 contra 0,2156 com 15 objetos a 6 px/quadro). Não é ruído nem acaso — o
    Hungarian maximiza a soma das IoU *do quadro*, que não é a quantidade que o IDF1 mede.
    Ser ótimo por quadro não é ser ótimo para a identidade, e isso é o assunto do PA2.
    """
    seqs = SyntheticVideos(n_sequences=3, seed=0, min_obj=15, max_obj=15, speed=5.0,
                           occlusion_duration=12, min_frames=40, max_frames=40)
    resultados = {}
    for regra in ("greedy", "hungarian"):
        tracker = IoUTracker(associacao=regra, iou_threshold=0.2, min_hits=1)
        total = 0.0
        for i in range(len(seqs)):
            s = seqs[i]
            dets = detections_from_sequence(s, min_visibility=0.0)
            total += evaluate(s, tracker.run(dets, s.name, s.fps, s.width, s.height)).idf1
        resultados[regra] = total / len(seqs)

    assert resultados["greedy"] != resultados["hungarian"]
    # nenhuma das duas é degenerada: as duas rastreiam de verdade
    assert all(0.2 < v < 0.9 for v in resultados.values())


def test_rastrear_a_mesma_entrada_duas_vezes_da_o_mesmo_resultado():
    """Sem determinismo, nenhuma comparação entre configurações significa coisa alguma."""
    seqs = SyntheticVideos(n_sequences=2, seed=1, min_obj=8, max_obj=8, speed=2.0,
                           occlusion_duration=10, min_frames=30, max_frames=30)
    s = seqs[0]
    dets = simulate_detections(s, drop_p=0.15, coord_noise=0.03, fp_rate=1.5, rng=5,
                               min_visibility=0.0)

    a = IoUTracker(min_hits=2).run(dets, s.name, s.fps, s.width, s.height)
    b = IoUTracker(min_hits=2).run(dets, s.name, s.fps, s.width, s.height)

    assert evaluate(s, a).to_dict() == evaluate(s, b).to_dict()


def test_o_rastreador_nao_carrega_estado_entre_sequencias():
    """Reusar o objeto em duas sequências não pode vazar tracks de uma para a outra."""
    a = sequencia({1: trilha(0, range(5))}, n_frames=5, nome="A")
    b = sequencia({1: trilha(0, range(5))}, n_frames=5, nome="B")

    tracker = IoUTracker(min_hits=1)
    tracker.run(detections_from_sequence(a), "A", 30.0, 1000, 1000)
    pred_b = tracker.run(detections_from_sequence(b), "B", 30.0, 1000, 1000)

    assert evaluate(b, pred_b).idf1 == 1.0
    assert len(pred_b.track_ids) == 1


def test_sequencia_sem_deteccao_nenhuma_nao_quebra():
    gabarito = sequencia({1: trilha(0, range(5))}, n_frames=5)
    vazias = [Detections(t, np.empty((0, 4)), np.empty(0)) for t in range(5)]
    predicao, m = _rastreia(gabarito, dets=vazias, min_hits=1)

    assert len(predicao) == 5
    assert m.idf1 == 0.0 and m.false_negatives == 5


def test_associacao_invalida_levanta_erro():
    with pytest.raises(ValueError, match="associacao"):
        IoUTracker(associacao="chute")
