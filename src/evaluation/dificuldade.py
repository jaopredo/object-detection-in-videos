"""Quanto uma sequência se move — o eixo de dificuldade da Parte 1.

O enunciado manda ordenar as sequências "pelo eixo de dificuldade que vocês escolherem:
densidade, movimento de câmera, duração de oclusão". Medimos os três e **densidade não
funciona**: nas 5 sequências de treino+validação ela correlaciona ao contrário com o
descolamento (r = −0,61). A MOT17-04 é a mais densa de todas (45 pedestres por quadro) e é
a que o baseline rastreia melhor (IDF1 0,717), porque é uma praça onde quase todo mundo está
parado ou andando devagar. Densidade mede quantos objetos há, não quanto eles se mexem — e o
que quebra associação por IoU é movimento.

O eixo usado aqui é o **deslocamento do centro da caixa entre quadros vizinhos, em unidades
da altura da própria caixa**. Duas propriedades importam:

*É livre de escala.* 10 px é muito para um pedestre de 30 px de altura e nada para um de
300. Sem normalizar, a MOT17-05 (640x480) pareceria a sequência mais parada do dataset só
por ser pequena — e ela é justamente uma das móveis.

*Mede o que a associação sofre.* Quando a câmera se mexe, o deslocamento aparente de um
objeto é movimento dele **mais** movimento da câmera, e o segundo não é previsível a partir
da caixa. É exatamente a informação que falta ao baseline, e o que a Parte 2 tem que
aprender a compensar.

Medido em 30/09: o eixo separa os dois grupos sem ambiguidade — câmeras paradas em 0,0035 e
0,0043, móveis em 0,0126, 0,0290 e 0,0597, um fator de 3 a 14. **Dentro** de cada grupo ele
não ordena o descolamento, e com 5 sequências não há como dizer se isso é falta de sinal ou
falta de amostra. Está dito assim no README, em vez de escolhido um eixo que ordenasse por
acaso.
"""

import numpy as np

from src.data.sequence import Sequence
from src.metrics.iou import iou_matrix


def deslocamento_relativo(sequence: Sequence) -> np.ndarray:
    """Deslocamento do centro de cada objeto entre quadros vizinhos, sobre a altura dele.

    Returns:
        (K,) uma amostra por (objeto, par de quadros consecutivos em que ele aparece nos
        dois). Vazio se nenhum objeto sobreviver a um par de quadros.
    """
    amostras = []
    for antes, depois in zip(sequence.frames[:-1], sequence.frames[1:]):
        comuns = np.intersect1d(antes.ids, depois.ids)
        if not len(comuns):
            continue
        onde_antes = {int(i): k for k, i in enumerate(antes.ids)}
        onde_depois = {int(i): k for k, i in enumerate(depois.ids)}
        for track_id in comuns:
            a = antes.boxes[onde_antes[int(track_id)]]
            b = depois.boxes[onde_depois[int(track_id)]]
            centro_a = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
            centro_b = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            altura = max(float(a[3] - a[1]), 1.0)
            amostras.append(float(np.linalg.norm(centro_b - centro_a) / altura))
    return np.array(amostras)


def iou_entre_quadros(sequence: Sequence) -> np.ndarray:
    """IoU entre a caixa de *t−1* e a de *t* **do mesmo objeto**.

    É literalmente o sinal de que a associação ingênua depende: se esta IoU cai abaixo do
    limiar do rastreador, nem um rastreador perfeito conseguiria casar as duas caixas. Serve
    de teto teórico do baseline, medido do gabarito e sem rodar nada.
    """
    amostras = []
    for antes, depois in zip(sequence.frames[:-1], sequence.frames[1:]):
        comuns = np.intersect1d(antes.ids, depois.ids)
        if not len(comuns):
            continue
        onde_antes = {int(i): k for k, i in enumerate(antes.ids)}
        onde_depois = {int(i): k for k, i in enumerate(depois.ids)}
        for track_id in comuns:
            a = antes.boxes[onde_antes[int(track_id)]:onde_antes[int(track_id)] + 1]
            b = depois.boxes[onde_depois[int(track_id)]:onde_depois[int(track_id)] + 1]
            amostras.append(float(iou_matrix(a, b)[0, 0]))
    return np.array(amostras)


def dificuldade(sequence: Sequence) -> dict:
    """Os eixos de dificuldade de uma sequência, medidos do gabarito."""
    passos = deslocamento_relativo(sequence)
    ious = iou_entre_quadros(sequence)
    por_quadro = [len(f) for f in sequence]

    return {
        "movimento": float(np.median(passos)) if len(passos) else 0.0,
        "movimento_p90": float(np.percentile(passos, 90)) if len(passos) else 0.0,
        "iou_vizinha": float(np.median(ious)) if len(ious) else 0.0,
        # fração de pares consecutivos em que nem um rastreador perfeito casaria com
        # limiar 0,3 — o teto do baseline, antes de rodar qualquer coisa
        "fora_de_alcance": float(np.mean(ious < 0.3)) if len(ious) else 0.0,
        "densidade": float(np.mean(por_quadro)) if por_quadro else 0.0,
    }
