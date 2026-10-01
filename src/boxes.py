"""Como a caixa entra e sai do modelo temporal — a decisão de projeto da Parte 2.

Vive na raiz de ``src/`` porque três pacotes dependem dela: ``data`` (recorta as janelas de
treino), ``models`` (a rede fala nesta linguagem) e ``tracking`` (o decodificador volta para
pixels). Pôr num deles obrigaria os outros dois a importar de um lugar que não é sobre eles.

## O que a rede prevê

Não a caixa, e sim o **incremento** — na mesma parametrização que o R-CNN usa para regressão
de caixa::

    tx = (cx' - cx) / w        deslocamento horizontal, em larguras da própria caixa
    ty = (cy' - cy) / h        deslocamento vertical, em alturas
    tw = log(w' / w)           mudança de escala, em log
    th = log(h' / h)

Duas escolhas estão embutidas aí, e as duas são o ponto.

**Incremento, não posição absoluta.** Prever a caixa absoluta deixaria a rede memorizar
*onde os pedestres costumam estar* em cada sequência do MOT17 — uma pista real, que funciona
na validação e desaparece numa câmera nova. Prevendo o incremento, ela é obrigada a aprender
*movimento*, que é o que generaliza.

**Dividido pelo tamanho da caixa, não pelo do quadro.** Um pedestre ao fundo anda 2 px por
quadro; o mesmo pedestre em primeiro plano anda 40. É o mesmo passo — o que mudou foi a
distância da câmera. Dividindo pela altura da caixa, os dois viram o mesmo alvo, e a rede
aprende uma vez o que de outro modo teria que aprender uma vez por profundidade.

É o que permite treinar em 1920x1080 e avaliar na MOT17-05, que é 640x480: medido, o
deslocamento mediano em pixels difere por ordens de grandeza entre as sequências, e em
unidades de altura da caixa fica na mesma faixa.

## O log na escala

``w'/w`` é uma razão: encolher pela metade é 0,5 e dobrar é 2,0. A distância de 1,0 não é a
mesma nos dois lados, e uma perda L1 sobre a razão puniria encolher menos do que crescer.
Em log, 0,5 e 2,0 ficam a ±0,69 — simétricos, que é o que a perda pressupõe.
"""

import numpy as np

#: piso de largura e altura, em pixels. Uma caixa de dimensão zero — que o ruído do
#: simulador ou uma detecção degenerada podem produzir — daria divisão por zero em ``tx`` e
#: ``log(0)`` em ``tw``, e o NaN resultante contamina o gradiente da janela inteira em
#: silêncio.
EPS = 1e-3


def xyxy_para_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """``(x1, y1, x2, y2)`` → ``(cx, cy, w, h)``."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w = np.maximum(boxes[:, 2] - boxes[:, 0], EPS)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], EPS)
    return np.stack([boxes[:, 0] + w / 2, boxes[:, 1] + h / 2, w, h], axis=1)


def cxcywh_para_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(cx, cy, w, h)`` → ``(x1, y1, x2, y2)``."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy = boxes[:, 0], boxes[:, 1]
    w = np.maximum(boxes[:, 2], EPS)
    h = np.maximum(boxes[:, 3], EPS)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def delta(de: np.ndarray, para: np.ndarray) -> np.ndarray:
    """O alvo da rede: o incremento que leva de uma caixa à outra.

    Args:
        de, para: (N, 4) em ``cxcywh``.

    Returns:
        (N, 4) com ``(tx, ty, tw, th)``.
    """
    de = np.asarray(de, dtype=np.float64).reshape(-1, 4)
    para = np.asarray(para, dtype=np.float64).reshape(-1, 4)
    w = np.maximum(de[:, 2], EPS)
    h = np.maximum(de[:, 3], EPS)
    return np.stack([
        (para[:, 0] - de[:, 0]) / w,
        (para[:, 1] - de[:, 1]) / h,
        np.log(np.maximum(para[:, 2], EPS) / w),
        np.log(np.maximum(para[:, 3], EPS) / h),
    ], axis=1)


def aplicar_delta(caixa: np.ndarray, d: np.ndarray) -> np.ndarray:
    """A inversa de ``delta``: aplica o incremento previsto e devolve a caixa nova.

    Args:
        caixa: (N, 4) em ``cxcywh``.
        d: (N, 4) com ``(tx, ty, tw, th)``.

    Returns:
        (N, 4) em ``cxcywh``.
    """
    caixa = np.asarray(caixa, dtype=np.float64).reshape(-1, 4)
    d = np.asarray(d, dtype=np.float64).reshape(-1, 4)
    w = np.maximum(caixa[:, 2], EPS)
    h = np.maximum(caixa[:, 3], EPS)
    # o expoente é limitado porque uma previsão ruim no começo do treino (tw = 20) viraria
    # uma caixa de 10^8 px, e daí NaN. ±4 já cobre encolher 50x e crescer 50x num passo.
    return np.stack([
        caixa[:, 0] + d[:, 0] * w,
        caixa[:, 1] + d[:, 1] * h,
        w * np.exp(np.clip(d[:, 2], -4.0, 4.0)),
        h * np.exp(np.clip(d[:, 3], -4.0, 4.0)),
    ], axis=1)


def contexto(caixa: np.ndarray, width: int, height: int) -> np.ndarray:
    """Onde a caixa está no quadro e que tamanho tem — o contexto absoluto da entrada.

    O incremento sozinho é cego à perspectiva: um pedestre no alto do quadro está longe e um
    embaixo está perto, e os dois se movem de formas diferentes mesmo depois de normalizar
    pelo tamanho. Estas quatro entradas dão à rede a chance de aprender essa diferença.

    Tudo dividido pelo tamanho do quadro, para que uma sequência 640x480 e uma 1920x1080
    cheguem na mesma faixa.

    Returns:
        (N, 4) com ``(cx/W, cy/H, w/W, h/H)``.
    """
    caixa = np.asarray(caixa, dtype=np.float64).reshape(-1, 4)
    return np.stack([
        caixa[:, 0] / width, caixa[:, 1] / height,
        caixa[:, 2] / width, caixa[:, 3] / height,
    ], axis=1)
