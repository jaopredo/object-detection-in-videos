"""O formato canônico do projeto: um vídeo anotado quadro a quadro.

Esta é a interface que separa *de onde os dados vêm* de *o que se faz com eles*. O gerador
sintético da Parte 0 e o leitor do MOT17 produzem ``Sequence``; métricas, tracker, figuras
e notebook consomem ``Sequence`` — e nenhum deles sabe qual das duas fontes está lendo.

É o que permite fechar e depurar a Parte 1 inteira no sintético antes de baixar os 5,5 GB
do MOT17, que é exatamente o que o enunciado propõe na Parte 0.

Duas convenções herdadas do MOT17 de propósito:

``boxes`` é **amodal**
    a caixa é a do objeto inteiro, mesmo quando parte dele está escondida. Um objeto
    totalmente ocluído continua na anotação, com ``visibility = 0``. É isso que permite
    verificar "sumiu por N quadros e voltou com a mesma identidade".

``visibility`` é a fração visível
    área que aparece na imagem dividida pela área que o objeto teria sozinho. O enunciado
    marca esse campo como "ouro para a Parte 4" — é ele que diz se o modelo errou porque a
    memória acabou ou porque o objeto nunca esteve lá.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """Um quadro e tudo que é verdade sobre ele.

    Attributes:
        index: posição na sequência, começando em 0.
        image: (H, W, 3) uint8. ``None`` quando só a geometria importa — o MOT17 permite
            trabalhar apenas com as anotações, sem carregar os JPEGs.
        boxes: (N, 4) float32 no formato ``xyxy`` (x1, y1, x2, y2), amodal.
        ids: (N,) int64. A identidade verdadeira, estável ao longo da sequência.
        visibility: (N,) float32 em [0, 1].
    """

    index: int
    boxes: np.ndarray
    ids: np.ndarray
    visibility: np.ndarray
    image: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        self.visibility = np.asarray(self.visibility, dtype=np.float32).reshape(-1)

        n = len(self.ids)
        if len(self.boxes) != n or len(self.visibility) != n:
            raise ValueError(
                f"quadro {self.index}: boxes ({len(self.boxes)}), ids ({n}) e "
                f"visibility ({len(self.visibility)}) precisam ter o mesmo comprimento"
            )
        if len(np.unique(self.ids)) != n:
            raise ValueError(
                f"quadro {self.index}: ids repetidos. Cada identidade aparece no máximo "
                f"uma vez por quadro — é a definição de identidade."
            )

    def __len__(self) -> int:
        return len(self.ids)

    def box_of(self, track_id: int) -> np.ndarray | None:
        """Caixa da identidade pedida neste quadro, ou ``None`` se ela não está aqui."""
        posicao = np.flatnonzero(self.ids == track_id)
        return self.boxes[posicao[0]] if len(posicao) else None


@dataclass
class Sequence:
    """Um vídeo anotado: metadados + a lista de quadros em ordem."""

    name: str
    fps: float
    width: int
    height: int
    frames: list[Frame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, indice: int) -> Frame:
        return self.frames[indice]

    def __iter__(self):
        return iter(self.frames)

    @property
    def track_ids(self) -> np.ndarray:
        """Todas as identidades que aparecem em algum quadro, ordenadas."""
        if not self.frames:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate([f.ids for f in self.frames]))

    def visibility_of(self, track_id: int) -> np.ndarray:
        """Série temporal de visibilidade de uma identidade, um valor por quadro.

        Quadros em que a identidade não está anotada contam como ``0.0``. A série tem
        sempre o comprimento da sequência, o que a torna direta de plotar e de varrer
        atrás do trecho contínuo de oclusão.
        """
        serie = np.zeros(len(self.frames), dtype=np.float32)
        for t, frame in enumerate(self.frames):
            posicao = np.flatnonzero(frame.ids == track_id)
            if len(posicao):
                serie[t] = frame.visibility[posicao[0]]
        return serie

    def has_images(self) -> bool:
        return bool(self.frames) and self.frames[0].image is not None
