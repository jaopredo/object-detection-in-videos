"""Os testes da métrica de identidade.

O enunciado exige três casos construídos à mão (Parte 0, artefato 3):

    (a) predição = gabarito              ⇒ IDF1 = 1 e zero switches
    (b) duas identidades trocadas em k   ⇒ o número exato de switches esperado
    (c) uma track partida em duas        ⇒ efeito em IDF1 diferente do de (b)

Cada um está abaixo com a conta feita à mão **no próprio teste**, não só o número final. Um
teste que afirma `idf1 == 0.7` sem mostrar de onde sai 0,7 não prova nada: se a
implementação e o teste tiverem o mesmo erro, os dois concordam.

É o análogo do teste "alimentar o gabarito como predição tem que dar mAP = 1,00" do PA1, e
tem o mesmo papel: a métrica é a única peça verificável sem treinar nada, e se ela estiver
errada todo número do trabalho está errado.
"""

import numpy as np
import pytest

from helpers.construtores import caixa, sequencia, trilha
from metrics import compute_idf1, evaluate, iou_matrix, match_greedy, match_hungarian


# =============================================================== (a) predição = gabarito

def test_a_predicao_igual_ao_gabarito_da_idf1_um():
    """Caso (a) do enunciado. O piso absoluto: se isto falha, nada mais importa."""
    gabarito = sequencia({1: trilha(0, range(10)),
                          2: trilha(1, range(10)),
                          3: trilha(2, range(10))}, n_frames=10)

    m = evaluate(gabarito, gabarito)

    assert m.idf1 == 1.0
    assert m.idp == 1.0 and m.idr == 1.0
    assert m.id_switches == 0
    assert m.fragmentations == 0
    assert m.count_error == 0
    assert m.mota == 1.0
    assert m.false_positives == 0 and m.false_negatives == 0
    # cada identidade verdadeira representada por ela mesma
    assert m.correspondencia == {1: 1, 2: 2, 3: 3}


def test_a_vale_para_um_gabarito_com_buracos():
    """O caso (a) não pode depender de todo objeto existir em todo quadro.

    Uma identidade que some no meio e volta é o assunto do PA2 inteiro; se a métrica
    perfeita já não fosse 1,0 aqui, ela estaria punindo o gabarito por ser o gabarito.
    """
    gabarito = sequencia({1: trilha(0, [0, 1, 2, 7, 8, 9]),   # some de 3 a 6 e volta
                          2: trilha(1, range(10))}, n_frames=10)

    m = evaluate(gabarito, gabarito)

    assert m.idf1 == 1.0
    assert m.id_switches == 0
    assert m.fragmentations == 0


# =========================================================== (b) duas identidades trocadas

def test_b_duas_identidades_trocadas_da_exatamente_dois_switches():
    """Caso (b) do enunciado: as duas previsões trocam de lugar no quadro 5.

    Conta à mão, 10 quadros, dois objetos parados em colunas distintas:

        overlap   pred1  pred2        quadros 0–4: pred1 na coluna 0, pred2 na coluna 1
        gt1 (c0)     5      5         quadros 5–9: pred1 na coluna 1, pred2 na coluna 0
        gt2 (c1)     5      5

    Todos os comprimentos são 10, então todo par custa 10 + 10 − 2·5 = 10: o Hungarian é
    indiferente e qualquer escolha dá IDTP = 5 + 5 = 10.

        IDF1 = 2·10 / (2·10 + 10 + 10) = 0,5

    Switches: no quadro 5 nenhum par anterior sobrevive (IoU 0 com quem era), o Hungarian
    recasa cruzado, e **as duas** identidades verdadeiras mudam de previsão. São 2 — e
    exatamente 2, porque dos quadros 6 em diante o par novo já é o do quadro anterior e
    fica preservado.
    """
    gabarito = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
    predicao = sequencia({
        1: {**trilha(0, range(0, 5)), **trilha(1, range(5, 10))},
        2: {**trilha(1, range(0, 5)), **trilha(0, range(5, 10))},
    }, n_frames=10)

    m = evaluate(gabarito, predicao)

    assert m.id_switches == 2
    assert m.idf1 == pytest.approx(0.5)
    assert m.idtp == 10 and m.idfp == 10 and m.idfn == 10
    # a troca não inventa nem perde identidade: a contagem continua certa
    assert m.n_pred_ids == 2 and m.count_error == 0
    # nenhuma caixa ficou sem par em nenhum quadro — o erro é só de rótulo
    assert m.false_positives == 0 and m.false_negatives == 0
    assert m.fragmentations == 0
    assert m.mota == pytest.approx(1.0 - 2 / 20)


def test_b_trocar_mais_tarde_nao_muda_a_contagem_de_switches():
    """A troca é um evento: acontece uma vez, independente de quando."""
    for k in (1, 5, 9):
        gabarito = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
        predicao = sequencia({
            1: {**trilha(0, range(0, k)), **trilha(1, range(k, 10))},
            2: {**trilha(1, range(0, k)), **trilha(0, range(k, 10))},
        }, n_frames=10)

        assert evaluate(gabarito, predicao).id_switches == 2, f"trocando no quadro {k}"


# ============================================================== (c) uma track partida

def test_c_track_partida_no_meio_difere_de_b_nas_tres_medidas():
    """Caso (c) do enunciado: **um** objeto, previsto como duas tracks.

    Conta à mão, 10 quadros, objeto parado, corte no quadro 3:

        pred1 cobre os quadros 0–2   (3 quadros)
        pred2 cobre os quadros 3–9   (7 quadros)

        custo(gt1, pred1) = 10 + 3 − 2·3 = 7
        custo(gt1, pred2) = 10 + 7 − 2·7 = 3   ← o Hungarian escolhe esta

        IDTP = 7 · IDFN = 10 − 7 = 3 · IDFP = 10 − 7 = 3
        IDF1 = 2·7 / (14 + 3 + 3) = 0,7

    É o ponto do enunciado: **partir não é trocar**. Contra o caso (b), as três medidas
    discordam — IDF1 0,7 contra 0,5; um switch contra dois; e aqui a contagem de
    identidades fica errada, o que em (b) não acontecia.
    """
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    predicao = sequencia({1: trilha(0, range(0, 3)),
                          2: trilha(0, range(3, 10))}, n_frames=10)

    m = evaluate(gabarito, predicao)

    assert m.idf1 == pytest.approx(0.7)
    assert m.idtp == 7 and m.idfp == 3 and m.idfn == 3
    assert m.id_switches == 1
    # a track partida inventa uma identidade que não existe — (b) não inventava nenhuma
    assert m.n_gt_ids == 1 and m.n_pred_ids == 2
    assert m.count_error == 1
    # e a representante oficial é o pedaço maior, não o primeiro
    assert m.correspondencia == {1: 2}


def test_c_partir_mais_vezes_derruba_mais_o_idf1():
    """Renumerar a cada quadro é o limite da fragmentação — e o IDF1 desaba.

    Um objeto de 10 quadros previsto como 10 tracks de 1 quadro: a melhor representante
    cobre um quadro só, então IDTP = 1 e IDF1 = 2/(2 + 9 + 9) = 0,1. Um olhar por quadro
    diria que este rastreador é perfeito — toda caixa está no lugar certo. É o que o
    enunciado quer que a métrica capture, e é a diferença entre instance-aware por quadro e
    identity-aware no tempo.
    """
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    predicao = sequencia({t + 1: trilha(0, [t]) for t in range(10)}, n_frames=10)

    m = evaluate(gabarito, predicao)

    assert m.idf1 == pytest.approx(0.1)
    assert m.id_switches == 9
    assert m.count_error == 9
    # zero FP e zero FN: por quadro, este rastreador não erra caixa nenhuma
    assert m.false_positives == 0 and m.false_negatives == 0


# ==================================================================== fragmentação

def test_fragmentacao_e_independente_de_switch():
    """Track interrompida e retomada **com o mesmo id**: 1 fragmentação, 0 switches.

    O gabarito existe nos 10 quadros; a predição cobre 0–3 e 7–9, com o mesmo id. É o
    padrão que um rastreador com ``max_age`` curto produz sob oclusão — e distingue as
    duas contagens: o rótulo nunca mudou (0 switches), mas a trajetória teve buraco.
    """
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    predicao = sequencia({1: trilha(0, [0, 1, 2, 3, 7, 8, 9])}, n_frames=10)

    m = evaluate(gabarito, predicao)

    assert m.fragmentations == 1
    assert m.id_switches == 0
    assert m.false_negatives == 3        # quadros 4, 5 e 6
    assert m.false_positives == 0
    assert m.idtp == 7 and m.idfn == 3 and m.idfp == 0
    assert m.idf1 == pytest.approx(2 * 7 / (2 * 7 + 0 + 3))


def test_track_que_some_e_nao_volta_nao_conta_fragmentacao():
    """Interrupção que nunca se recupera não é fragmentação — é a track acabando.

    A definição está escrita em ``compute_clearmot``: contamos **retomadas**. Sem isso, um
    objeto que sai de cena no fim do vídeo penalizaria o rastreador por tê-lo perdido
    exatamente quando ele deixou de existir.
    """
    gabarito = sequencia({1: trilha(0, range(10))}, n_frames=10)
    predicao = sequencia({1: trilha(0, range(0, 5))}, n_frames=10)

    m = evaluate(gabarito, predicao)

    assert m.fragmentations == 0
    assert m.id_switches == 0
    assert m.false_negatives == 5


# ============================================================ o casamento que preserva

def test_cruzamento_sem_troca_de_id_nao_inventa_switch():
    """Dois objetos que se cruzam e cada um mantém seu id: zero switches.

    Este é o teste que justifica o ``match_clearmot``. No quadro do encontro as duas caixas
    quase coincidem, e um casamento decidido do zero poderia cruzar os pares por causa de
    uma diferença de IoU na terceira casa — contando dois switches que ninguém cometeu. A
    fase que preserva o par anterior impede isso.
    """
    # os dois se aproximam, encostam no quadro 2 e se afastam
    esquerda = {0: [0, 0, 40, 40], 1: [30, 0, 70, 40], 2: [55, 0, 95, 40],
                3: [80, 0, 120, 40], 4: [110, 0, 150, 40]}
    direita = {0: [150, 0, 190, 40], 1: [120, 0, 160, 40], 2: [58, 0, 98, 40],
               3: [40, 0, 80, 40], 4: [0, 0, 40, 40]}

    gabarito = sequencia({1: esquerda, 2: direita}, n_frames=5)
    m = evaluate(gabarito, gabarito)

    assert m.id_switches == 0
    assert m.idf1 == 1.0


# ==================================================================== casos degenerados

def test_sequencia_vazia_dos_dois_lados_e_idf1_um():
    vazia = sequencia({}, n_frames=5)
    m = evaluate(vazia, vazia)
    assert m.idf1 == 1.0 and m.mota == 1.0 and m.id_switches == 0


def test_nao_prever_nada_da_idf1_zero():
    gabarito = sequencia({1: trilha(0, range(5))}, n_frames=5)
    m = evaluate(gabarito, sequencia({}, n_frames=5))

    assert m.idf1 == 0.0
    assert m.idfn == 5 and m.idfp == 0
    assert m.false_negatives == 5
    assert m.count_error == 1


def test_prever_lixo_onde_nao_ha_nada_da_idf1_zero_e_mota_negativa():
    """MOTA pode ficar abaixo de zero, e isso não é bug: ela não é uma fração de acerto."""
    vazia = sequencia({}, n_frames=5)
    predicao = sequencia({1: trilha(0, range(5)), 2: trilha(1, range(5))}, n_frames=5)
    m = evaluate(vazia, predicao)

    assert m.idf1 == 0.0
    assert m.idfp == 10
    assert m.mota == 1.0       # sem caixa verdadeira não há por que dividir: convenção

    # com uma caixa verdadeira só, os 10 falsos positivos afundam a MOTA
    gabarito = sequencia({9: trilha(5, [0])}, n_frames=5)
    assert evaluate(gabarito, predicao).mota < 0


def test_comprimentos_diferentes_levantam_erro():
    """Comparar 10 quadros com 5 daria um IDF1 bom porque metade do vídeo sumiu."""
    with pytest.raises(ValueError, match="quadros"):
        evaluate(sequencia({1: trilha(0, range(10))}, n_frames=10),
                 sequencia({1: trilha(0, range(5))}, n_frames=5))


# ========================================================================= IoU e regras

def test_iou_de_caixas_conhecidas():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10],      # idêntica
                  [5, 0, 15, 10],      # metade: interseção 50, união 150
                  [20, 20, 30, 30]],   # disjunta
                 dtype=np.float32)

    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[0, 2] == 0.0


def test_iou_com_conjunto_vazio_devolve_matriz_vazia_e_nao_erro():
    vazio = np.empty((0, 4), dtype=np.float32)
    cheio = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(vazio, cheio).shape == (0, 1)
    assert iou_matrix(cheio, vazio).shape == (1, 0)
    assert iou_matrix(vazio, vazio).shape == (0, 0)


def test_iou_de_caixa_degenerada_e_zero_e_nao_nan():
    """O ruído do simulador pode gerar caixa de área nula; um nan envenenaria o Hungarian."""
    degenerada = np.array([[5, 5, 5, 5]], dtype=np.float32)
    normal = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = iou_matrix(degenerada, normal)
    assert iou[0, 0] == 0.0 and not np.isnan(iou).any()


def test_guloso_e_hungarian_discordam_no_caso_construido():
    """O caso mínimo em que a melhor escolha local impede duas escolhas boas.

              x     y
        A   0,9   0,6
        B   0,8   0,0

    O guloso pega A–x (o maior de todos) e sobra para B só o y, que está abaixo do limiar:
    **um** par, somando 0,9. O Hungarian abre mão de A–x e leva A–y + B–x: **dois** pares,
    somando 1,4. É o que o enunciado quer dizer com "regras diferentes dão números
    diferentes" — e por isso a escolha está documentada no README.
    """
    iou = np.array([[0.9, 0.6], [0.8, 0.0]])

    assert match_greedy(iou, 0.5) == [(0, 0)]
    assert match_hungarian(iou, 0.5) == [(0, 1), (1, 0)]


def test_as_duas_regras_concordam_quando_nao_ha_conflito():
    iou = np.array([[0.9, 0.1], [0.1, 0.8]])
    assert match_greedy(iou, 0.5) == match_hungarian(iou, 0.5) == [(0, 0), (1, 1)]


def test_nenhuma_regra_casa_abaixo_do_limiar():
    iou = np.array([[0.4, 0.3], [0.2, 0.49]])
    assert match_greedy(iou, 0.5) == []
    assert match_hungarian(iou, 0.5) == []


def test_idf1_nao_depende_da_ordem_dos_ids():
    """Renomear as identidades previstas não pode mudar o número.

    IDF1 é definida sobre uma atribuição global; se ela dependesse de qual id recebeu qual
    número, estaria medindo a numeração e não o rastreamento.
    """
    gabarito = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
    normal = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
    renomeada = sequencia({77: trilha(0, range(10)), 3: trilha(1, range(10))}, n_frames=10)

    assert compute_idf1(gabarito, normal).idf1 == compute_idf1(gabarito, renomeada).idf1 == 1.0


# ====================================================== verificação independente do IDF1

def _idf1_por_forca_bruta(gt, pred, threshold=0.5):
    """IDF1 calculada enumerando **todas** as atribuições possíveis.

    Existe para conferir o Hungarian com um algoritmo diferente, não para ser usada: é
    fatorial e só serve em casos minúsculos. Um erro na montagem da matriz quadrada com
    preenchimento — o pedaço mais fácil de errar de ``compute_idf1``, e o mais silencioso,
    porque uma matriz errada devolve um número plausível — aparece aqui na hora.

    A definição usada é a mesma, sem compartilhar código nenhum: conta as coexistências,
    procura a atribuição parcial um-para-um de maior IDTP, e aplica a fórmula.
    """
    from itertools import combinations, permutations

    ids_gt, ids_pred = list(gt.track_ids), list(pred.track_ids)
    if not ids_gt and not ids_pred:
        return 1.0
    if not ids_gt or not ids_pred:
        return 0.0

    overlap = {(g, p): 0 for g in ids_gt for p in ids_pred}
    total_gt = total_pred = 0
    for qg, qp in zip(gt, pred):
        total_gt += len(qg)
        total_pred += len(qp)
        for i, g in enumerate(qg.ids):
            for j, p in enumerate(qp.ids):
                if iou_matrix(qg.boxes[i:i + 1], qp.boxes[j:j + 1])[0, 0] >= threshold:
                    overlap[(int(g), int(p))] += 1

    # melhor atribuição parcial: escolhe um subconjunto de cada lado e um emparelhamento
    melhor = 0
    k_max = min(len(ids_gt), len(ids_pred))
    for k in range(k_max + 1):
        for sub_gt in combinations(ids_gt, k):
            for sub_pred in permutations(ids_pred, k):
                melhor = max(melhor, sum(
                    overlap[(int(g), int(p))] for g, p in zip(sub_gt, sub_pred)
                ))

    return 2 * melhor / (total_gt + total_pred) if (total_gt + total_pred) else 1.0


@pytest.mark.parametrize("semente", range(12))
def test_idf1_bate_com_a_forca_bruta_em_casos_aleatorios(semente):
    """Cenários sorteados: o Hungarian tem que dar o mesmo que a enumeração exaustiva."""
    rng = np.random.RandomState(semente)
    n_quadros = 8

    def sorteia(n_tracks):
        return {
            i + 1: {
                t: caixa(int(rng.randint(0, 4)), int(rng.randint(0, 3)))
                for t in range(n_quadros) if rng.rand() < 0.75
            }
            for i in range(n_tracks)
        }

    gabarito = sequencia(sorteia(int(rng.randint(1, 4))), n_frames=n_quadros)
    predicao = sequencia(sorteia(int(rng.randint(1, 5))), n_frames=n_quadros)

    assert compute_idf1(gabarito, predicao).idf1 == pytest.approx(
        _idf1_por_forca_bruta(gabarito, predicao)
    )


def test_forca_bruta_confere_os_tres_casos_do_enunciado():
    """O verificador independente reproduz 1,0 · 0,5 · 0,7 dos casos (a), (b) e (c)."""
    a = sequencia({1: trilha(0, range(10)), 2: trilha(1, range(10))}, n_frames=10)
    assert _idf1_por_forca_bruta(a, a) == pytest.approx(1.0)

    b = sequencia({
        1: {**trilha(0, range(0, 5)), **trilha(1, range(5, 10))},
        2: {**trilha(1, range(0, 5)), **trilha(0, range(5, 10))},
    }, n_frames=10)
    assert _idf1_por_forca_bruta(a, b) == pytest.approx(0.5)

    gt_c = sequencia({1: trilha(0, range(10))}, n_frames=10)
    pred_c = sequencia({1: trilha(0, range(0, 3)), 2: trilha(0, range(3, 10))}, n_frames=10)
    assert _idf1_por_forca_bruta(gt_c, pred_c) == pytest.approx(0.7)
