"""Regras de casamento entre dois conjuntos de caixas num mesmo quadro.

O enunciado do PA1 já dizia, e o do PA2 repete: *regras diferentes dão números diferentes*,
e a escolha tem que ser explícita. Por isso as três regras que o projeto usa moram juntas
aqui, com a diferença entre elas escrita — em vez de uma delas estar embutida no meio de uma
métrica, onde ninguém a encontraria para discutir.

    guloso       pega o melhor par disponível, repete. Rápido, e localmente ótimo apenas.
    Hungarian    maximiza a soma das IoU do quadro inteiro. Globalmente ótimo no quadro.
    CLEAR-MOT    Hungarian, mas **preservando o casamento do quadro anterior** quando ele
                 ainda vale.

A terceira é a que a contagem de ID switches exige, e é onde está a sutileza do PA2. Se o
casamento de cada quadro for decidido do zero, dois objetos que se cruzam trocam de par por
conta da geometria e a métrica conta um switch que o rastreador não cometeu — a métrica
estaria medindo a si mesma. Preservar a correspondência anterior enquanto ela for válida faz
o switch contado ser o que o rastreador de fato fez.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_hungarian(iou: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Casamento um-para-um que maximiza a soma das IoU do quadro.

    Args:
        iou: (N, M) matriz de IoU.
        threshold: pares com IoU **estritamente menor** que isto não são casados.

    Returns:
        Lista de pares ``(i, j)``, ordenada por ``i``.
    """
    if iou.size == 0:
        return []

    # linear_sum_assignment minimiza; queremos maximizar IoU. Pares abaixo do limiar
    # recebem custo alto o bastante para nunca compensarem — mas finito: o scipy recusa
    # `inf` na matriz, e um `nan` daria uma atribuição arbitrária sem reclamar.
    custo = np.where(iou >= threshold, -iou, 1.0)
    linhas, colunas = linear_sum_assignment(custo)

    return [
        (int(i), int(j)) for i, j in zip(linhas, colunas) if iou[i, j] >= threshold
    ]


def match_greedy(iou: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Casamento guloso: repetidamente pega o par de maior IoU ainda disponível.

    Difere do Hungarian quando a melhor escolha local impede duas escolhas boas. O caso
    mínimo, construído no teste: A-x = 0,9, A-y = 0,6, B-x = 0,8, B-y = 0. O guloso pega
    A-x (0,9) e sobra B-y (0), somando 0,9; o Hungarian pega A-y + B-x, somando 1,4.
    """
    if iou.size == 0:
        return []

    pares: list[tuple[int, int]] = []
    livres_i = np.ones(iou.shape[0], dtype=bool)
    livres_j = np.ones(iou.shape[1], dtype=bool)

    # uma ordenação só, no começo: reordenar a cada passo seria O(n² log n) sem ganho,
    # já que a IoU de um par não muda quando outro par é escolhido
    achatado = np.argsort(-iou, axis=None)
    for indice in achatado:
        i, j = np.unravel_index(indice, iou.shape)
        if iou[i, j] < threshold:
            break  # a partir daqui tudo é pior, porque a lista está ordenada
        if livres_i[i] and livres_j[j]:
            pares.append((int(i), int(j)))
            livres_i[i] = livres_j[j] = False

    return sorted(pares)


def match_clearmot(
    iou: np.ndarray,
    gt_ids: np.ndarray,
    pred_ids: np.ndarray,
    ultimo_par: dict[int, int],
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Casamento do quadro que **preserva a correspondência do quadro anterior**.

    É a regra da família CLEAR-MOT, e a única com que a contagem de ID switches significa
    alguma coisa. Duas fases:

    1. toda identidade verdadeira que já estava casada com uma previsão, e cujo par ainda
       está neste quadro com IoU suficiente, **mantém o par** — mesmo que exista outra
       previsão com IoU maior agora;
    2. o que sobrou dos dois lados vai para o Hungarian.

    A fase 1 é o ponto. Sem ela, dois pedestres que se cruzam trocam de par por geometria e
    a métrica contaria dois switches que ninguém cometeu.

    Args:
        iou: (N, M) IoU entre as caixas verdadeiras e as previstas deste quadro.
        gt_ids: (N,) identidades verdadeiras, na ordem das linhas de ``iou``.
        pred_ids: (M,) identidades previstas, na ordem das colunas.
        ultimo_par: identidade verdadeira → última identidade prevista a que esteve casada.
            Acumulado ao longo da sequência pelo chamador; não é modificado aqui.
        threshold: IoU mínima para um par valer.

    Returns:
        Lista de pares ``(i, j)`` de **índices** (não de ids), ordenada por ``i``.
    """
    if iou.size == 0:
        return []

    posicao_pred = {int(p): j for j, p in enumerate(pred_ids)}
    pares: list[tuple[int, int]] = []
    livres_i = np.ones(len(gt_ids), dtype=bool)
    livres_j = np.ones(len(pred_ids), dtype=bool)

    # fase 1 — manter o que já estava casado
    for i, g in enumerate(gt_ids):
        j = posicao_pred.get(ultimo_par.get(int(g), -1), -1)
        if j >= 0 and iou[i, j] >= threshold:
            pares.append((i, j))
            livres_i[i] = livres_j[j] = False

    # fase 2 — Hungarian no que sobrou
    resto_i = np.flatnonzero(livres_i)
    resto_j = np.flatnonzero(livres_j)
    if len(resto_i) and len(resto_j):
        for i, j in match_hungarian(iou[np.ix_(resto_i, resto_j)], threshold):
            pares.append((int(resto_i[i]), int(resto_j[j])))

    return sorted(pares)
