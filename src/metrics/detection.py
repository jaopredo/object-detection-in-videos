"""Qualidade da **detecção**, separada da qualidade da identidade.

O gráfico obrigatório da Parte 1 tem dois painéis: em cima mAP, embaixo identidades e ID
switches. O argumento inteiro do PA2 está na **distância entre os dois** — a detecção vai
bem e a identidade não —, e para esse argumento existir é preciso medir as duas coisas com
métricas que não conversam entre si. Daí este módulo ser separado de ``identity.py``: aqui
não existe o conceito de identidade, só caixa e score.

Como a fonte de detecções fica congelada a partir da Parte 2, este número é **constante** ao
longo de todo o resto do trabalho. É exatamente isso que o torna útil: qualquer variação de
IDF1 entre a Parte 1 e a Parte 5 não pode ser atribuída à detecção, porque ela não mudou.
"""

import numpy as np

from src.data.detections import Detections
from src.data.sequence import Sequence
from src.metrics.iou import iou_matrix


def average_precision(
    detections: list[Detections],
    gt: Sequence,
    threshold: float = 0.5,
    min_visibility: float | None = None,
) -> dict:
    """AP@``threshold`` de uma sequência inteira, pela área sob a curva precisão-revocação.

    A convenção, que o enunciado exige que seja explícita:

    - as detecções de **todos** os quadros entram numa lista só, ordenada por score
      decrescente. Não é a média das APs por quadro: um quadro com 2 objetos pesaria tanto
      quanto um com 60, e no MOT17 essa diferença é a sequência inteira;
    - dentro de cada quadro, cada detecção casa com a caixa verdadeira livre de maior IoU
      (guloso por score, que é a convenção do PASCAL VOC e do COCO). Caixa verdadeira já
      casada não casa de novo — a segunda detecção em cima dela é falso positivo;
    - a AP é a **área sob a curva**, com todos os pontos, sem a interpolação em 11 pontos do
      VOC antigo.

    Args:
        detections: uma lista por quadro, na ordem dos quadros.
        gt: o gabarito, com os mesmos quadros.
        threshold: IoU mínima para uma detecção contar como acerto.
        min_visibility: ignora objetos verdadeiros com visibilidade até este valor. Um
            objeto totalmente ocluído não pode ser detectado, e contá-lo como falso negativo
            mediria a oclusão da cena em vez do detector.

    Returns:
        ``{"ap", "precision", "recall", "n_det", "n_gt", "tp", "fp"}``.
    """
    if len(detections) != len(gt):
        raise ValueError(
            f"{len(detections)} quadros de detecção e {len(gt)} de gabarito"
        )

    scores: list[float] = []
    acertos: list[bool] = []
    n_verdadeiras = 0

    for dets, frame in zip(detections, gt):
        visivel = (
            slice(None) if min_visibility is None else frame.visibility > min_visibility
        )
        caixas_gt = frame.boxes[visivel]
        n_verdadeiras += len(caixas_gt)
        if len(dets) == 0:
            continue

        ordem = np.argsort(-dets.scores)
        iou = iou_matrix(dets.boxes[ordem], caixas_gt)
        livres = np.ones(len(caixas_gt), dtype=bool)

        for posicao in range(len(ordem)):
            scores.append(float(dets.scores[ordem[posicao]]))
            if not len(caixas_gt):
                acertos.append(False)
                continue
            candidatas = np.where(livres, iou[posicao], -1.0)
            melhor = int(np.argmax(candidatas))
            if candidatas[melhor] >= threshold:
                livres[melhor] = False
                acertos.append(True)
            else:
                acertos.append(False)

    if not scores or n_verdadeiras == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0,
                "n_det": len(scores), "n_gt": n_verdadeiras, "tp": 0, "fp": len(scores)}

    ordem_global = np.argsort(-np.array(scores), kind="stable")
    acertos = np.array(acertos, dtype=bool)[ordem_global]

    tp = np.cumsum(acertos)
    fp = np.cumsum(~acertos)
    precisao = tp / (tp + fp)
    revocacao = tp / n_verdadeiras

    # área sob a curva, com a precisão tornada monótona decrescente da direita para a
    # esquerda: sem isso, o serrilhado da curva de precisão (que sobe a cada acerto) daria
    # uma AP que depende de onde caiu cada empate de score
    precisao_max = np.maximum.accumulate(precisao[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], revocacao])) * precisao_max))

    return {
        "ap": ap,
        "precision": float(precisao[-1]),
        "recall": float(revocacao[-1]),
        "n_det": int(len(acertos)),
        "n_gt": int(n_verdadeiras),
        "tp": int(tp[-1]),
        "fp": int(fp[-1]),
    }
