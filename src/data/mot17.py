"""Leitor do MOT17 e o split por sequência.

O MOT17 chega como 21 pastas em ``train/``: 7 sequências × 3 detectores públicos. As três
cópias de uma sequência **têm o mesmo ``gt.txt``** (conferido por md5) — o que muda é só o
``det.txt``. Então "escolher o detector" é escolher qual arquivo de detecção ler, e não
muda nada do gabarito.

O ``test/`` do benchmark não tem ``gt.txt`` (o gabarito ficou com os organizadores), então
**o nosso split inteiro sai do ``train/``**. É a mesma situação do ``stage1_test`` do
DSB2018 no PA1.

Formato, medido em 30/09 (e não suposto — o enunciado descreve o ``gt.txt`` sem mencionar
que a sétima coluna é um flag)::

    gt.txt   frame, id, x, y, w, h, flag(0|1), classe, visibility
    det.txt  frame,  -1, x, y, w, h, conf                            (só 7 campos)

Ver ``read_mot_gt(only_active_pedestrians=True)`` para o que o flag e a classe descartam.
"""

from dataclasses import dataclass
from pathlib import Path

from src.data.detections import Detections, read_mot_det
from src.data.sequence import Sequence
from src.data.mot_format import read_mot_gt

#: os três detectores públicos que acompanham o MOT17.
DETECTORES = ("DPM", "FRCNN", "SDP")


@dataclass(frozen=True)
class InfoSequencia:
    """O que se sabe de uma sequência antes de abrir o gabarito.

    ``camera`` e ``densidade`` são os eixos de dificuldade que o enunciado sugere para
    ordenar as sequências nos gráficos da Parte 1 e para justificar o split.
    """

    nome: str
    camera: str        #: "parada" ou "móvel"
    n_frames: int
    fps: float
    width: int
    height: int
    n_ids: int
    densidade: float   #: média de pedestres considerados por quadro


#: Medido em 30/09 com o pacote de anotações; a câmera vem da descrição do benchmark.
#: Serve para ordenar as sequências e para justificar o split **sem precisar abrir os
#: arquivos** — o que deixa o README e as figuras coerentes mesmo antes de ler o dataset.
SEQUENCIAS: dict[str, InfoSequencia] = {
    "MOT17-02": InfoSequencia("MOT17-02", "parada", 600, 30.0, 1920, 1080, 62, 31.0),
    "MOT17-04": InfoSequencia("MOT17-04", "parada", 1050, 30.0, 1920, 1080, 83, 45.3),
    "MOT17-05": InfoSequencia("MOT17-05", "móvel", 837, 14.0, 640, 480, 133, 8.3),
    "MOT17-09": InfoSequencia("MOT17-09", "parada", 525, 30.0, 1920, 1080, 26, 10.1),
    "MOT17-10": InfoSequencia("MOT17-10", "móvel", 654, 30.0, 1920, 1080, 57, 19.6),
    "MOT17-11": InfoSequencia("MOT17-11", "móvel", 900, 30.0, 1920, 1080, 75, 10.5),
    "MOT17-13": InfoSequencia("MOT17-13", "móvel", 750, 25.0, 1920, 1080, 110, 15.5),
}

#: Split **por sequência**, nunca por quadro. O enunciado é explícito: separar quadros ao
#: acaso põe o quadro t no treino e o t+1 na validação, e aí o modelo temporal é avaliado
#: em cima de algo que ele praticamente já viu.
#:
#: O critério, para a apresentação:
#:
#: - **treino** cobre os dois tipos de câmera e a faixa inteira de densidade (10 a 45).
#:   Sem a 04 (a mais densa) o modelo nunca veria multidão; sem a 11 e a 13 nunca veria
#:   câmera em movimento, que é o que quebra um modelo de movimento aprendido em pixels.
#: - **validação** é a 10: móvel, densidade no meio da faixa. Escolha de época e de
#:   hiperparâmetro acontece só aqui.
#: - **teste** são duas, e por motivos diferentes. A 09 é parada e esparsa — o caso fácil,
#:   que serve de piso. A **05 está fora da distribuição em dois eixos ao mesmo tempo**:
#:   640x480 (as outras seis são 1920x1080) e 14 fps (as outras são 25 ou 30). É a
#:   sequência que mede se o modelo aprendeu movimento ou decorou a escala do MOT17 — e
#:   conversa direto com a Parte 5, que é sobre taxa de quadros.
SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("MOT17-02", "MOT17-04", "MOT17-11", "MOT17-13"),
    "val": ("MOT17-10",),
    "test": ("MOT17-09", "MOT17-05"),
}

#: treino + validação, para as figuras que precisam de **várias** sequências no eixo x.
#: O gráfico do descolamento da Parte 1 ordena as sequências por dificuldade, e com uma
#: sequência só ele não mostra nada. Existe para que essa figura não seja desculpa para
#: abrir o conjunto de teste antes da hora.
SPLITS["trainval"] = SPLITS["train"] + SPLITS["val"]


def pasta_da_sequencia(root: str | Path, nome: str, detector: str = "SDP") -> Path:
    """Caminho de ``<root>/train/MOT17-XX-DET``, com erro útil se não existir."""
    if detector not in DETECTORES:
        raise ValueError(f"detector {detector!r} não existe. Use um de {DETECTORES}.")

    destino = Path(root) / "train" / f"{nome}-{detector}"
    if not destino.is_dir():
        raise FileNotFoundError(
            f"{destino} não existe. Baixe as anotações do MOT17 com `make dados` "
            f"(ou veja o README). Sequências conhecidas: {sorted(SEQUENCIAS)}."
        )
    return destino


def ler_seqinfo(pasta: Path) -> dict:
    """Lê o ``seqinfo.ini`` — fps, tamanho do quadro e nº de quadros."""
    valores = {}
    for linha in (pasta / "seqinfo.ini").read_text(encoding="utf-8").splitlines():
        if "=" in linha:
            chave, valor = linha.split("=", 1)
            valores[chave.strip()] = valor.strip()
    return {
        "fps": float(valores["frameRate"]),
        "n_frames": int(valores["seqLength"]),
        "width": int(valores["imWidth"]),
        "height": int(valores["imHeight"]),
        "im_dir": valores.get("imDir", "img1"),
        "im_ext": valores.get("imExt", ".jpg"),
    }


def read_mot17_gt(root: str | Path, nome: str, detector: str = "SDP") -> Sequence:
    """O gabarito de uma sequência, já filtrado para pedestres considerados.

    O ``detector`` não muda nada aqui — o ``gt.txt`` das três cópias é idêntico. Está na
    assinatura só para o chamador não precisar saber disso.
    """
    pasta = pasta_da_sequencia(root, nome, detector)
    info = ler_seqinfo(pasta)
    return read_mot_gt(
        pasta / "gt" / "gt.txt",
        name=nome,
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        n_frames=info["n_frames"],
        only_active_pedestrians=True,
    )


def read_mot17_det(
    root: str | Path, nome: str, detector: str = "SDP", min_score: float | None = None
) -> list[Detections]:
    """As detecções públicas de uma sequência.

    Args:
        min_score: descarta detecções abaixo deste score. ``None`` mantém tudo — o DPM
            entrega scores em outra escala dos outros dois, então um limiar único
            significaria coisas diferentes em cada detector.
    """
    pasta = pasta_da_sequencia(root, nome, detector)
    info = ler_seqinfo(pasta)
    dets = read_mot_det(pasta / "det" / "det.txt", n_frames=info["n_frames"])
    return [d.filter_score(min_score) for d in dets] if min_score is not None else dets


def sequencias_do_split(split: str) -> tuple[str, ...]:
    """Os nomes das sequências de um split, com erro útil se o nome estiver errado."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r} não existe. Use um de {sorted(SPLITS)}.")
    return SPLITS[split]


def caminho_do_quadro(root: str | Path, nome: str, quadro: int,
                      detector: str = "SDP") -> Path:
    """Caminho do JPEG de um quadro (0-indexado em memória, 1-indexado em disco).

    As imagens são idênticas nas três variantes de detector — só o ``det.txt`` muda —, então
    só uma cópia precisa estar no disco. É por isso que a extração do pacote de 5,86 GB pega
    apenas ``MOT17-*-SDP/img1``: 857 MB em vez de 2,6 GB, sem perder nada.
    """
    pasta = pasta_da_sequencia(root, nome, detector)
    info = ler_seqinfo(pasta)
    return pasta / info["im_dir"] / f"{quadro + 1:06d}{info['im_ext']}"


def ler_quadro(root: str | Path, nome: str, quadro: int, detector: str = "SDP"):
    """Lê um quadro como array ``(H, W, 3)`` uint8, ou ``None`` se a imagem não foi baixada.

    Devolver ``None`` em vez de levantar erro é deliberado: as Partes 0 a 3 e 5 funcionam só
    com as anotações (10 MB), e o projeto inteiro tem que continuar rodando para quem não
    baixou os 5,86 GB. Quem precisa de imagem — a galeria da Parte 4 e o notebook — avisa.
    """
    import imageio.v2 as imageio

    caminho = caminho_do_quadro(root, nome, quadro, detector)
    return imageio.imread(caminho) if caminho.exists() else None


class MOT17Split:
    """Um split do MOT17 como coleção de ``Sequence``, carregada sob demanda.

    Mesma interface do ``SyntheticVideos`` da Parte 0 — ``len`` e ``__getitem__`` devolvendo
    ``Sequence`` — de modo que métricas, rastreador e figuras não distingam sintético de
    MOT17. É o que o gerador da Parte 0 foi desenhado para permitir, e é o que fez a Parte 1
    inteira ser depurada antes de o download dos 5,86 GB terminar.

    Carrega sob demanda porque o gabarito da 04 tem 47 mil linhas: a ablação da Parte 3
    instancia este objeto dezenas de vezes e quase nunca lê todas as sequências.
    """

    def __init__(self, root: str | Path, split: str, detector: str = "SDP"):
        self.root = Path(root)
        self.split = split
        self.detector = detector
        self.nomes = sequencias_do_split(split)
        self._cache: dict[str, Sequence] = {}

    def __len__(self) -> int:
        return len(self.nomes)

    def __getitem__(self, indice: int) -> Sequence:
        nome = self.nomes[indice]
        if nome not in self._cache:
            self._cache[nome] = read_mot17_gt(self.root, nome, self.detector)
        return self._cache[nome]

    def detections(self, indice: int) -> list[Detections]:
        return read_mot17_det(self.root, self.nomes[indice], self.detector)

    def info(self, indice: int) -> InfoSequencia:
        return SEQUENCIAS[self.nomes[indice]]

    def __repr__(self) -> str:
        return f"MOT17Split({self.split}, {self.detector}, {list(self.nomes)})"
