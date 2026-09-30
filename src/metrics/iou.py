"""Interseção sobre união entre conjuntos de caixas.

Vive em ``metrics/`` e não em ``tracking/`` de propósito: IoU é uma **medição**, e tanto o
rastreador quanto as métricas dependem dela. Pondo aqui, a seta de dependência aponta numa
direção só — ``tracking`` importa de ``metrics``, nunca o contrário — e a peça mais
verificada do projeto não depende da menos verificada.

Vetorizada por broadcast, não por laço aninhado. É a lição medida do PA1: a matriz de IoU
entre 80 objetos numa imagem 256x256 caiu de 1.528 ms para 0,89 ms quando deixou de ser
dois ``for`` (1712x). Aqui importa pelo mesmo motivo — a Parte 3 roda 36 avaliações
completas, cada uma varrendo milhares de quadros.
"""

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz ``(N, M)`` de IoU entre dois conjuntos de caixas ``xyxy``.

    Args:
        a: (N, 4) caixas no formato ``(x1, y1, x2, y2)``.
        b: (M, 4) idem.

    Returns:
        (N, M) float64, com ``saida[i, j]`` = IoU entre ``a[i]`` e ``b[j]``. Conjunto vazio
        de um dos lados devolve uma matriz de forma ``(N, 0)`` ou ``(0, M)`` — nunca um
        erro, porque quadro sem detecção é situação normal, não excepcional.

    Note:
        União zero (caixa degenerada, de área nula) devolve IoU 0, não ``nan``. Uma caixa
        de área zero pode aparecer depois do ruído do simulador, e um ``nan`` solto na
        matriz envenenaria o Hungarian em silêncio — ele devolveria uma atribuição
        arbitrária em vez de levantar erro.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    intersecao = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    uniao = area_a[:, None] + area_b[None, :] - intersecao

    return np.where(uniao > 0, intersecao / np.where(uniao > 0, uniao, 1.0), 0.0)
