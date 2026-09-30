"""Detecções: o que um detector devolve, sem identidade nenhuma.

Este é o tipo que falta para a Parte 1 existir. O projeto já tinha ``Frame``, que é uma
anotação — caixa **mais identidade**, com a garantia de que cada identidade aparece no
máximo uma vez por quadro. Detecção não é isso: o detector devolve um conjunto de caixas
com score, sem ordem canônica e sem nome. É o primeiro parágrafo do enunciado:

    "o conjunto do quadro t não tem relação nenhuma com o conjunto do quadro t+1: a ordem
    das saídas vem do NMS, que ordena por score."

Tentar reaproveitar ``Frame`` para isso quebraria na hora, porque ``Frame.__post_init__``
exige ids únicos e o ``det.txt`` do MOT17 traz ``id = -1`` em toda linha. O tipo separado
não é burocracia: é o que torna impossível confundir, numa assinatura de função, o que já
tem identidade com o que ainda precisa ganhar uma. Dar identidade a isto é o trabalho
inteiro do PA2.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.sequence import Frame, Sequence


@dataclass
class Detections:
    """As caixas que um detector devolveu num quadro.

    Attributes:
        frame_index: posição na sequência, começando em 0 — mesma convenção de ``Frame``.
        boxes: (N, 4) float32 no formato ``xyxy``.
        scores: (N,) float32. Confiança do detector. O ``gt.txt`` usado como fonte de
            detecção entra com score 1,0 em tudo.
    """

    frame_index: int
    boxes: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if len(self.boxes) != len(self.scores):
            raise ValueError(
                f"quadro {self.frame_index}: boxes ({len(self.boxes)}) e scores "
                f"({len(self.scores)}) precisam ter o mesmo comprimento"
            )

    def __len__(self) -> int:
        return len(self.scores)

    def top_k(self, k: int) -> "Detections":
        """As ``k`` detecções de maior score. ``k >= N`` devolve tudo, reordenado."""
        ordem = np.argsort(-self.scores)[:k]
        return Detections(self.frame_index, self.boxes[ordem], self.scores[ordem])

    def filter_score(self, minimo: float) -> "Detections":
        """Descarta o que está abaixo do limiar de confiança."""
        mantem = self.scores >= minimo
        return Detections(self.frame_index, self.boxes[mantem], self.scores[mantem])


def detections_from_sequence(
    sequence: Sequence, min_visibility: float | None = None
) -> list[Detections]:
    """As caixas do gabarito servidas como se fossem detecções perfeitas.

    É o detector-oráculo: score 1,0 em tudo, nada inventado. Serve de teto para a Parte 1 —
    com detecção perfeita, todo erro que sobrar é de associação, que é o único assunto do
    PA2.

    Args:
        sequence: a sequência de onde tirar as caixas.
        min_visibility: descarta objetos com visibilidade **menor ou igual** a isto.
            ``None`` (padrão) não descarta nada.

    Note:
        ``min_visibility=0.0`` é quase sempre o que se quer, e a razão merece estar escrita
        porque custou um experimento que não mediu nada.

        A caixa do gabarito é **amodal**: ela cobre o objeto inteiro mesmo quando parte
        dele — ou ele todo — está escondido atrás de outra coisa. Isso é uma propriedade da
        *anotação*, não do que existe na imagem. Um objeto com ``visibility = 0`` não põe um
        único pixel na tela, e nenhum detector do mundo devolve uma caixa para ele.

        Servir o gabarito amodal como detecção cria um detector que **enxerga através das
        coisas**. Com ele, girar o botão de duração da oclusão no gerador sintético não
        muda nada: medimos IDF1 = 1,0000 para oclusões de 0, 5, 10 e 20 quadros, porque a
        oclusão simplesmente não chegava ao rastreador. O oráculo tem que ser cego ao que
        está escondido; só o que ele vê é que pode ser associado.
    """
    saida = []
    for f in sequence:
        visivel = (
            slice(None) if min_visibility is None else f.visibility > min_visibility
        )
        boxes = f.boxes[visivel]
        saida.append(Detections(f.index, boxes.copy(), np.ones(len(boxes), dtype=np.float32)))
    return saida


def detections_to_sequence(
    detections: list[Detections], name: str, fps: float, width: int, height: int,
) -> Sequence:
    """Empacota detecções numa ``Sequence`` com ids provisórios ``0..N-1`` por quadro.

    Os ids **não são identidades** — são só posições dentro do quadro, renumeradas do zero
    a cada quadro. Existe para reaproveitar as figuras e o escritor de MOT, que falam
    ``Sequence``. Nunca passe o resultado disto para uma métrica de identidade: ela mediria
    um rastreador que troca o id de todo mundo a cada quadro.
    """
    frames = [
        Frame(
            index=d.frame_index,
            boxes=d.boxes,
            ids=np.arange(len(d)),
            visibility=np.ones(len(d), dtype=np.float32),
        )
        for d in detections
    ]
    return Sequence(name=name, fps=fps, width=width, height=height, frames=frames)


def read_mot_det(path: str | Path, n_frames: int | None = None) -> list[Detections]:
    """Lê um ``det.txt`` do MOTChallenge.

    O formato é o mesmo do ``gt.txt`` nas seis primeiras colunas e diverge depois::

        gt.txt   frame, id, bb_left, bb_top, bb_width, bb_height, flag, classe, visibility
        det.txt  frame, -1, bb_left, bb_top, bb_width, bb_height, conf

    Duas diferenças que obrigam a um leitor próprio, em vez de reusar ``read_mot_gt``:

    - **``id`` é sempre −1.** Passar isso para ``Frame`` levanta erro de id repetido no
      primeiro quadro com duas detecções — e levanta com razão: detecção não tem
      identidade;
    - **a coluna 7 é confiança de verdade**, um número contínuo. No ``gt.txt`` a mesma
      posição é um *flag* de 0 ou 1 dizendo se a linha entra na avaliação. Mesma coluna,
      significados diferentes; tratar as duas com o mesmo código é como se erra.

    Args:
        path: caminho do ``det.txt``.
        n_frames: comprimento da sequência. Sem isto, usa o maior quadro visto no arquivo —
            o que subestima quando os últimos quadros não têm detecção nenhuma.

    Returns:
        Uma lista com **um item por quadro**, inclusive os quadros sem detecção. O índice
        na lista é o índice do quadro: ``dets[t]`` é sempre o quadro ``t``.
    """
    path = Path(path)
    por_quadro: dict[int, list[tuple]] = {}
    maior = -1

    for linha in path.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        campos = linha.split(",")
        quadro = int(float(campos[0])) - 1  # 1-indexado em disco → 0-indexado em memória
        left, top, w, h = (float(c) for c in campos[2:6])
        conf = float(campos[6]) if len(campos) > 6 else 1.0
        por_quadro.setdefault(quadro, []).append((left, top, left + w, top + h, conf))
        maior = max(maior, quadro)

    total = n_frames if n_frames is not None else maior + 1
    saida = []
    for t in range(total):
        registros = por_quadro.get(t, [])
        if registros:
            dados = np.array(registros, dtype=np.float32)
            saida.append(Detections(t, dados[:, :4], dados[:, 4]))
        else:
            saida.append(Detections(t, np.empty((0, 4)), np.empty(0)))
    return saida


def write_mot_det(detections: list[Detections], path: str | Path) -> Path:
    """Grava detecções em ``det.txt``, para cachear a saída de um detector caro.

    O detector do torchvision da Parte 1 leva minutos por sequência em CPU. Gravando o
    resultado neste formato, ele vira indistinguível de uma detecção pública para o resto
    do projeto — e não precisa rodar de novo em nenhuma das partes seguintes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    linhas = []
    for d in detections:
        for (x1, y1, x2, y2), score in zip(d.boxes, d.scores):
            linhas.append(
                f"{d.frame_index + 1},-1,{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f}"
            )

    path.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    return path
