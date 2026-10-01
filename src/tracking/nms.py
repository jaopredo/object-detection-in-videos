"""Supressão de não-máximos — implementação nossa.

O enunciado proíbe ``torchvision.ops.nms`` em letras claras: *"vocês implementam o NMS"*.

O NMS é também o lugar de onde vem o problema do PA2 inteiro, e vale dizer isso aqui porque
é fácil ver o algoritmo como um detalhe de detecção. O primeiro parágrafo do enunciado:

    "o conjunto do quadro t não tem relação nenhuma com o conjunto do quadro t+1: a ordem
    das saídas vem do NMS, que ordena por score. O objeto que era o terceiro da lista vira
    o primeiro quando alguém passa na frente dele."

O NMS é por quadro e ordena por confiança. A posição de um objeto na lista de saída não tem
memória nenhuma — é exatamente a ausência que o modelo temporal da Parte 2 vai preencher.
"""

import numpy as np

from src.metrics.iou import iou_matrix


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Índices das caixas que sobrevivem à supressão, em ordem de score decrescente.

    O algoritmo padrão: pega a caixa de maior score, elimina toda caixa que se sobrepõe a
    ela acima do limiar, repete com o que sobrou.

    Args:
        boxes: (N, 4) no formato ``xyxy``.
        scores: (N,) confiança.
        iou_threshold: sobreposição **acima** da qual a caixa de menor score é suprimida.

    Returns:
        (K,) índices em ``boxes``, ordenados por score decrescente.

    Note:
        A IoU contra a caixa escolhida é calculada de uma vez contra **todas** as
        candidatas restantes, não uma a uma. A versão com laço aninhado é o mesmo erro que
        o PA1 mediu na matriz de IoU (1712x mais lenta) — e aqui doeria na Parte 1, que roda
        o NMS em 5.000 quadros de MOT17 por fonte de detecção.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError(f"boxes ({len(boxes)}) e scores ({len(scores)}) têm tamanhos diferentes")
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    ordem = np.argsort(-scores, kind="stable")   # estável: empate resolve pelo índice, e o
    mantidos = []                                # resultado não depende do algoritmo de sort

    while len(ordem):
        melhor = ordem[0]
        mantidos.append(int(melhor))
        if len(ordem) == 1:
            break
        ious = iou_matrix(boxes[melhor:melhor + 1], boxes[ordem[1:]])[0]
        ordem = ordem[1:][ious <= iou_threshold]

    return np.array(mantidos, dtype=np.int64)


def nms_por_classe(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """NMS independente dentro de cada classe.

    Duas classes diferentes ocupando o mesmo lugar não são detecção duplicada — é um
    pedestre em cima de uma bicicleta. Só entra em jogo no detector do torchvision, que
    devolve as 80 classes do COCO; a Parte 1 filtra *person* antes e cai no NMS simples.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    mantidos = []
    for c in np.unique(classes):
        onde = np.flatnonzero(classes == c)
        mantidos.extend(onde[nms(boxes[onde], np.asarray(scores)[onde], iou_threshold)])
    # reordena globalmente por score: o chamador espera a mesma convenção do `nms`
    mantidos = np.array(sorted(mantidos), dtype=np.int64)
    return mantidos[np.argsort(-np.asarray(scores)[mantidos], kind="stable")]
