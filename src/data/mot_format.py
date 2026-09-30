"""Leitura e escrita do formato MOTChallenge.

O enunciado descreve o ``gt.txt`` do MOT17 assim::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

O gerador sintético da Parte 0 grava **neste mesmo formato**. Não é capricho: é o que faz
com que o resto do projeto — métricas, tracker, figuras, notebook — não precise distinguir
sintético de MOT17. Um bug de associação descoberto no sintético é o mesmo bug que
apareceria no MOT17, e sai muito mais barato de encontrar.

Duas conversões acontecem aqui e em nenhum outro lugar do projeto:

    - ``frame`` é 1-indexado em disco e 0-indexado em memória (``Frame.index``);
    - a caixa é ``xywh`` (canto superior esquerdo + tamanho) em disco e ``xyxy`` em memória.

Concentrar as duas neste módulo evita o erro clássico de ficar somando ou subtraindo 1 em
lugares espalhados até os números baterem por acidente.
"""

from pathlib import Path

import numpy as np

from src.data.sequence import Frame, Sequence

#: classe "pedestre" do MOT17. O sintético usa a mesma para não inventar um vocabulário
#: paralelo; nada no projeto filtra por classe ainda.
PEDESTRE = 1


def write_mot_gt(sequence: Sequence, path: str | Path) -> Path:
    """Grava as anotações de uma sequência em ``gt.txt``.

    Args:
        sequence: a sequência a serializar.
        path: caminho do arquivo. Diretórios intermediários são criados.

    Returns:
        O caminho gravado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    linhas = []
    for frame in sequence:
        for box, track_id, vis in zip(frame.boxes, frame.ids, frame.visibility):
            x1, y1, x2, y2 = box
            linhas.append(
                f"{frame.index + 1},{int(track_id)},"
                f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                f"1,{PEDESTRE},{vis:.4f}"
            )

    path.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    return path


def read_mot_gt(
    path: str | Path,
    name: str | None = None,
    fps: float = 30.0,
    width: int = 0,
    height: int = 0,
    n_frames: int | None = None,
) -> Sequence:
    """Lê um ``gt.txt`` e devolve a ``Sequence`` correspondente (sem imagens).

    Args:
        path: caminho do ``gt.txt``.
        name: nome da sequência. Se ``None``, usa o nome da pasta que contém o arquivo.
        fps, width, height: metadados que o ``gt.txt`` não carrega. O MOT17 os traz num
            ``seqinfo.ini`` separado; aqui entram por parâmetro.
        n_frames: comprimento da sequência. Se ``None``, usa o maior número de quadro
            visto no arquivo — o que subestima quando os últimos quadros estão vazios.

    Returns:
        A sequência, com um ``Frame`` por índice, inclusive os quadros sem nenhum objeto.
    """
    path = Path(path)
    texto = path.read_text(encoding="utf-8").strip()
    if name is None:
        name = path.parent.name

    por_quadro: dict[int, list[tuple]] = {}
    maior_quadro = -1
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        campos = linha.split(",")
        quadro = int(float(campos[0])) - 1  # 1-indexado em disco → 0-indexado em memória
        track_id = int(float(campos[1]))
        left, top, w, h = (float(c) for c in campos[2:6])
        # `visibility` é opcional: um det.txt vai só até `conf`. Sem o campo, assume-se
        # totalmente visível — é o que o resto do projeto espera de uma detecção.
        vis = float(campos[8]) if len(campos) > 8 else 1.0
        por_quadro.setdefault(quadro, []).append(
            (left, top, left + w, top + h, track_id, vis)
        )
        maior_quadro = max(maior_quadro, quadro)

    total = n_frames if n_frames is not None else maior_quadro + 1

    frames = []
    for indice in range(total):
        registros = por_quadro.get(indice, [])
        if registros:
            dados = np.array(registros, dtype=np.float64)
            boxes, ids, vis = dados[:, :4], dados[:, 4], dados[:, 5]
        else:
            boxes = np.empty((0, 4))
            ids = np.empty(0)
            vis = np.empty(0)
        frames.append(Frame(index=indice, boxes=boxes, ids=ids, visibility=vis))

    return Sequence(name=name, fps=fps, width=width, height=height, frames=frames)
