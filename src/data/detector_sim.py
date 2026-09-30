"""Simulador de detector — artefato 2 da Parte 0.

Recebe as caixas verdadeiras e as estraga de propósito, do jeito que o enunciado descreve:
descarta ``p%`` delas, adiciona ruído nas coordenadas, injeta falsos positivos. Com isso dá
para **validar a Parte 1 inteira no sintético**, onde a resposta certa é conhecida, antes de
encostar no MOT17.

Vale notar por que isto é mais do que um utilitário de teste: é o mesmo experimento da
Parte 5 na variante "qualidade do detector", e é como se constrói o caso (c) dos testes da
métrica (uma track partida no meio). Três usos, um módulo.

**Os três botões são independentes de propósito.** Um detector real erra de formas
correlacionadas — perde justamente o que está ocluído, e inventa justamente onde há
textura. Misturar isso aqui tornaria impossível responder "qual dos três defeitos quebra a
associação?", que é a pergunta que o simulador existe para responder. A correlação com
oclusão já existe no gerador sintético, por construção; aqui ela ficaria de fora.

Tudo é função pura de um ``RandomState``: a mesma seed dá exatamente as mesmas detecções
estragadas, o que é o que permite comparar duas regras de associação sobre a mesma entrada
ruim.
"""

import numpy as np

from src.data.detections import Detections
from src.data.sequence import Sequence

#: faixa de score dos verdadeiros positivos e dos falsos positivos. Elas se **sobrepõem**
#: de propósito: um simulador que desse 0,9 a todo verdadeiro e 0,1 a todo falso tornaria o
#: limiar de confiança um filtro perfeito, e aí o experimento não mediria nada. A
#: sobreposição é o que faz o score ser uma pista útil mas insuficiente — como num detector
#: de verdade.
FAIXA_SCORE_TP = (0.5, 1.0)
FAIXA_SCORE_FP = (0.1, 0.6)


def simulate_detections(
    sequence: Sequence,
    drop_p: float = 0.0,
    coord_noise: float = 0.0,
    fp_rate: float = 0.0,
    rng: np.random.RandomState | int | None = None,
    min_visibility: float | None = None,
) -> list[Detections]:
    """Estraga as caixas do gabarito e devolve o resultado como detecções.

    Args:
        sequence: a sequência cujas caixas verdadeiras serão estragadas.
        drop_p: fração das caixas verdadeiras descartada, em [0, 1]. Vira falso negativo.
        coord_noise: desvio do ruído gaussiano nas coordenadas, **em fração da diagonal da
            própria caixa**. Relativo, não absoluto: 3 px de erro é desprezível num
            pedestre de 300 px de altura e é o objeto inteiro num de 10 px. Um σ absoluto
            faria o botão significar coisas diferentes em cada sequência do MOT17, que vão
            de 640x480 a 1920x1080.
        fp_rate: número **esperado** de falsos positivos por quadro, amostrado de uma
            Poisson. Não é uma fração: contar FP como "p% a mais" amarraria a quantidade de
            lixo à quantidade de objetos, e um detector ruim inventa coisas num quadro vazio
            do mesmo jeito.
        rng: ``RandomState``, inteiro de seed, ou ``None`` (não-determinístico).
        min_visibility: descarta objetos escondidos **antes** de qualquer outro defeito —
            ver a nota em ``detections_from_sequence``. Este descarte não é um defeito do
            detector: é o que qualquer detector faria, porque não há pixel para detectar.
            Os três botões modelam o detector; este modela a cena.

    Returns:
        Uma lista com um ``Detections`` por quadro, na ordem dos quadros.

    Note:
        Com ``drop_p = coord_noise = fp_rate = 0`` o resultado é o gabarito **bit a bit**,
        com scores 1,0 — nenhum número aleatório é consumido. É o teste de identidade do
        módulo, e é o que garante que o piso do experimento é o detector perfeito.
    """
    if not 0.0 <= drop_p <= 1.0:
        raise ValueError(f"drop_p tem que estar em [0, 1], não {drop_p}")
    if coord_noise < 0 or fp_rate < 0:
        raise ValueError("coord_noise e fp_rate não podem ser negativos")

    rng = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)
    perfeito = drop_p == 0.0 and coord_noise == 0.0 and fp_rate == 0.0
    tamanhos = _tamanhos_tipicos(sequence)

    saida = []
    for frame in sequence:
        visivel = (
            slice(None) if min_visibility is None else frame.visibility > min_visibility
        )
        caixas_visiveis = frame.boxes[visivel]

        if perfeito:
            saida.append(Detections(
                frame.index, caixas_visiveis.copy(),
                np.ones(len(caixas_visiveis), dtype=np.float32),
            ))
            continue

        boxes = caixas_visiveis.copy()
        mantidas = rng.rand(len(boxes)) >= drop_p
        boxes = boxes[mantidas]

        if coord_noise > 0 and len(boxes):
            boxes = _perturbar(boxes, coord_noise, rng)

        scores = rng.uniform(*FAIXA_SCORE_TP, size=len(boxes)).astype(np.float32)

        if fp_rate > 0:
            falsas = _falsos_positivos(
                int(rng.poisson(fp_rate)), tamanhos, sequence.width, sequence.height, rng
            )
            if len(falsas):
                boxes = np.concatenate([boxes, falsas])
                scores = np.concatenate([
                    scores,
                    rng.uniform(*FAIXA_SCORE_FP, size=len(falsas)).astype(np.float32),
                ])

        # ordena por score decrescente: é a ordem em que um detector de verdade entrega,
        # porque é a ordem em que o NMS cospe. E é justamente a ordem que o enunciado
        # aponta como enganosa — "o objeto que era o terceiro da lista vira o primeiro
        # quando alguém passa na frente dele". Entregar já ordenado evita que algum código
        # nosso passe a depender, sem perceber, da ordem do gabarito.
        ordem = np.argsort(-scores)
        saida.append(Detections(frame.index, boxes[ordem], scores[ordem]))

    return saida


def _perturbar(boxes: np.ndarray, coord_noise: float, rng) -> np.ndarray:
    """Soma ruído gaussiano nas 4 coordenadas, com σ proporcional à diagonal da caixa."""
    larguras = boxes[:, 2] - boxes[:, 0]
    alturas = boxes[:, 3] - boxes[:, 1]
    sigma = coord_noise * np.hypot(larguras, alturas)

    ruido = rng.normal(0.0, 1.0, boxes.shape) * sigma[:, None]
    saida = (boxes + ruido).astype(np.float32)

    # o ruído pode inverter os cantos numa caixa pequena, e uma caixa com x2 < x1 tem IoU
    # negativa com tudo — um bug que se propaga silencioso até a métrica. Reordenar é mais
    # honesto que descartar: a detecção existe, só saiu torta.
    x1 = np.minimum(saida[:, 0], saida[:, 2])
    x2 = np.maximum(saida[:, 0], saida[:, 2])
    y1 = np.minimum(saida[:, 1], saida[:, 3])
    y2 = np.maximum(saida[:, 1], saida[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _tamanhos_tipicos(sequence: Sequence) -> np.ndarray:
    """Larguras e alturas observadas na sequência inteira, para dimensionar os FPs.

    Falso positivo com tamanho sorteado uniformemente no quadro seria fácil demais de
    filtrar — bastaria uma regra de área. Amostrando do que existe de fato na cena, o lixo
    fica com a mesma cara dos objetos, e o único jeito de rejeitá-lo é temporal: ele não
    persiste. Que é exatamente o que o modelo da Parte 2 deveria conseguir fazer.
    """
    tamanhos = [
        np.stack([f.boxes[:, 2] - f.boxes[:, 0], f.boxes[:, 3] - f.boxes[:, 1]], axis=1)
        for f in sequence if len(f)
    ]
    if tamanhos:
        return np.concatenate(tamanhos)
    # sequência sem nenhum objeto: um palpite de 10% do quadro, para o simulador continuar
    # funcionando em vez de quebrar num caso de borda
    return np.array([[0.1 * sequence.width, 0.1 * sequence.height]], dtype=np.float32)


def _falsos_positivos(n: int, tamanhos, width: int, height: int, rng) -> np.ndarray:
    """``n`` caixas inventadas, com tamanho amostrado dos objetos reais da sequência."""
    if n <= 0:
        return np.empty((0, 4), dtype=np.float32)

    escolhidos = tamanhos[rng.randint(0, len(tamanhos), size=n)]
    w, h = escolhidos[:, 0], escolhidos[:, 1]
    x1 = rng.uniform(0, np.maximum(width - w, 1))
    y1 = rng.uniform(0, np.maximum(height - h, 1))
    return np.stack([x1, y1, x1 + w, y1 + h], axis=1).astype(np.float32)
