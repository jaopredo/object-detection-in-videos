"""Construtores de sequências pequenas e previsíveis, para os testes de métrica.

Os testes da métrica precisam de casos em que a resposta certa é calculável à mão. Estes
helpers deixam o caso caber em três linhas, para que o teste mostre o **cenário** e não a
mecânica de montar arrays.
"""

import numpy as np

from src.data.sequence import Frame, Sequence


def caixa(coluna: int, linha: int = 0, lado: int = 40) -> list[float]:
    """Uma caixa quadrada numa grade grossa. Colunas distintas não se tocam."""
    x, y = 100 * coluna, 100 * linha
    return [x, y, x + lado, y + lado]


def sequencia(
    tracks: dict[int, dict[int, list]],
    n_frames: int,
    nome: str = "TESTE",
    fps: float = 30.0,
    width: int = 1000,
    height: int = 1000,
) -> Sequence:
    """Monta uma ``Sequence`` a partir de ``{identidade: {quadro: caixa}}``.

    Quadros sem nenhum objeto viram ``Frame`` vazios, não somem — é o que o resto do
    projeto espera, e é o que faz o índice na lista ser o índice do quadro.
    """
    frames = []
    for t in range(n_frames):
        ids, boxes = [], []
        for track_id, por_quadro in sorted(tracks.items()):
            if t in por_quadro:
                ids.append(track_id)
                boxes.append(por_quadro[t])
        frames.append(Frame(
            index=t,
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            ids=np.array(ids, dtype=np.int64),
            visibility=np.ones(len(ids), dtype=np.float32),
        ))
    return Sequence(name=nome, fps=fps, width=width, height=height, frames=frames)


def trilha(coluna: int, quadros) -> dict[int, list]:
    """Um objeto parado na coluna dada, presente nos quadros pedidos."""
    return {t: caixa(coluna) for t in quadros}
