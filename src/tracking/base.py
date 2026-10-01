"""O que todo rastreador do projeto tem em comum: tracks, gestão de nascimento e morte.

O enunciado pede (Parte 1, item 4) que a regra de associação e a gestão de nascimento/morte
de tracks estejam **documentadas**, e avisa que regras diferentes dão números diferentes.
Elas moram aqui, num lugar só, porque o rastreador ingênuo da Parte 1 e o recorrente da
Parte 2 têm que compartilhar exatamente estas regras: se cada um gerisse tracks do seu
jeito, a comparação entre os dois mediria as duas coisas ao mesmo tempo e não diria nada
sobre a recorrência, que é o objeto do trabalho.

O que difere entre os dois é só **de onde sai a caixa prevista** para o quadro seguinte:

    IoUTracker   a última caixa observada (modelo de movimento nenhum: o objeto está onde
                 estava)
    RNNTracker   a caixa que o estado recorrente prevê

Todo o resto — o limiar, o Hungarian, quando nasce, quando morre — é o mesmo código.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.data.detections import Detections
from src.data.sequence import Frame, Sequence
from src.metrics.iou import iou_matrix
from src.metrics.matching import match_greedy, match_hungarian


@dataclass
class Track:
    """Um objeto sendo seguido, com o estado que sobrevive entre quadros."""

    id: int
    box: np.ndarray            #: última caixa conhecida ou prevista, ``xyxy``
    score: float
    nasceu_em: int             #: quadro em que a track foi criada
    visto_em: int              #: último quadro com observação de verdade
    hits: int = 1              #: quantas observações já recebeu
    age: int = 0               #: quadros seguidos **sem** observação
    #: caixas ainda não emitidas, à espera de confirmação — ver ``min_hits`` em
    #: ``TrackManager``. Pares ``(quadro, caixa)``.
    pendentes: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def confirmada(self) -> bool:
        return not self.pendentes


class TrackManager:
    """Nascimento, morte e confirmação de tracks — as regras, num lugar só.

    Args:
        iou_threshold: IoU mínima entre a caixa prevista e a detecção para elas casarem.
            Abaixo disto a detecção é tratada como objeto novo.
        max_age: quantos quadros seguidos sem observação uma track sobrevive antes de
            morrer. **É o botão que a Parte 4 mede:** o horizonte de memória empírico é
            quantos quadros o estado aguenta antes da track morrer ou trocar de id, e este
            número é o teto duro disso. Com ``max_age`` menor que a oclusão típica do
            dataset, nenhum modelo temporal por melhor que seja consegue atravessá-la.
        min_hits: quantas observações uma track precisa antes de aparecer na saída. Filtra
            o falso positivo isolado do detector, que de outro modo viraria uma identidade
            inteira e estragaria a contagem de identidades únicas.
        min_score: descarta detecções abaixo deste score antes de qualquer associação.
        associacao: ``"hungarian"`` (ótimo no quadro) ou ``"greedy"``. O enunciado pede que
            a escolha seja explícita; ver ``src/metrics/matching.py`` para o caso construído
            em que as duas discordam.
        emitir_sem_observacao: se uma track sem detecção neste quadro mesmo assim entra na
            saída, com a caixa prevista. ``False`` no rastreador ingênuo, cuja "previsão" é
            a caixa velha parada; ``True`` no recorrente, para o qual atravessar a oclusão é
            o ponto.

    Note:
        ``min_hits`` emite **retroativamente**. Quando a track é confirmada, as caixas que
        ficaram guardadas são despejadas nos quadros a que pertencem. A alternativa — jogar
        fora os primeiros quadros de toda track — custaria ``min_hits - 1`` quadros de
        cada identidade em IDF1, de graça, já que a avaliação é offline e esses quadros
        estão ali. O preço é latência, que só existiria num sistema online.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        min_score: float = 0.0,
        associacao: str = "hungarian",
        emitir_sem_observacao: bool = False,
    ):
        if associacao not in ("hungarian", "greedy"):
            raise ValueError(
                f"associacao {associacao!r} não existe. Use 'hungarian' ou 'greedy'."
            )
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_score = min_score
        self.associacao = associacao
        self.emitir_sem_observacao = emitir_sem_observacao

    @property
    def casar(self):
        return match_hungarian if self.associacao == "hungarian" else match_greedy

    def config(self) -> dict:
        """As regras como dicionário, para irem junto do resultado no JSON."""
        return {
            "iou_threshold": self.iou_threshold, "max_age": self.max_age,
            "min_hits": self.min_hits, "min_score": self.min_score,
            "associacao": self.associacao,
            "emitir_sem_observacao": self.emitir_sem_observacao,
        }


class Tracker(ABC):
    """Contrato de um rastreador: recebe detecções quadro a quadro, devolve identidades.

    A subclasse implementa duas coisas — como prever a caixa de cada track no próximo
    quadro, e o que fazer com o estado quando chega (ou não chega) uma observação. O laço
    que percorre a sequência, associa, cria, mata e monta a saída é o mesmo para todas, e
    está em ``run``.
    """

    def __init__(self, manager: TrackManager | None = None, **kwargs):
        self.manager = manager or TrackManager(**kwargs)
        self.reset()

    def reset(self) -> None:
        """Zera o estado. Chamado entre sequências — uma track nunca atravessa vídeos."""
        self.tracks: list[Track] = []
        self._proximo_id = 1
        self._saida: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.width = self.height = 0

    # ------------------------------------------------------------ o que a subclasse faz

    @abstractmethod
    def prever(self, quadro: int) -> np.ndarray:
        """Caixas previstas para ``quadro``, uma por track viva, na ordem de ``self.tracks``.

        Returns:
            (N, 4) ``xyxy``. ``N == len(self.tracks)``.
        """

    @abstractmethod
    def observar(self, track: Track, box: np.ndarray, score: float, quadro: int) -> None:
        """Absorve uma detecção que casou com esta track."""

    @abstractmethod
    def sem_observacao(self, track: Track, quadro: int) -> None:
        """Avança a track um quadro sem detecção — é aqui que a oclusão acontece."""

    @abstractmethod
    def nascer(self, track: Track) -> None:
        """Prepara o estado interno de uma track recém-criada."""

    def fim_do_quadro(self, quadro: int, dt: float) -> None:
        """Chamado depois de todas as tracks terem sido atualizadas neste quadro.

        Existe para o trabalho que vale a pena fazer **em lote**. O ``RNNTracker`` roda a
        recorrência aqui, com uma linha por track viva, numa chamada só: um quadro do
        MOT17-04 tem 60 tracks, e sessenta chamadas de ``GRUCell`` com uma linha cada
        custariam muito mais que uma chamada com sessenta.

        O padrão não faz nada — o rastreador ingênuo não tem o que rodar.
        """

    # ------------------------------------------------------------------- o laço comum

    def run(
        self, detections: list[Detections], name: str, fps: float, width: int, height: int,
        stride: int = 1,
    ) -> Sequence:
        """Percorre a sequência inteira e devolve as trajetórias como ``Sequence``.

        A saída é uma ``Sequence`` — o mesmo tipo do gabarito — de modo que ``metrics.py``
        compare as duas sem saber qual é qual. É a mesma decisão que fez a Parte 0 gravar em
        formato MOTChallenge.

        Args:
            stride: quantos quadros do vídeo original cada passo desta lista representa.
                Vale 1 no caso normal e 2 ou 5 na Parte 5, que subamostra o vídeo. É o que
                faz ``dt`` chegar certo na recorrência: com ``stride = 5`` cada passo cobre
                cinco vezes mais tempo, e um modelo de movimento que não souber disso vai
                prever um quinto do deslocamento.
        """
        self.reset()
        m = self.manager
        self.width, self.height = width, height
        dt = stride / fps if fps else 1.0

        for dets in detections:
            t = dets.frame_index
            self._saida.setdefault(t, [])

            if m.min_score > 0:
                dets = dets.filter_score(m.min_score)

            previstas = self.prever(t)
            pares = m.casar(iou_matrix(previstas, dets.boxes), m.iou_threshold)

            casadas_track = {i for i, _ in pares}
            casadas_det = {j for _, j in pares}

            for i, j in pares:
                self.observar(self.tracks[i], dets.boxes[j], float(dets.scores[j]), t)

            for i, track in enumerate(self.tracks):
                if i not in casadas_track:
                    self.sem_observacao(track, t)

            for j in range(len(dets)):
                if j not in casadas_det:
                    self._criar(dets.boxes[j], float(dets.scores[j]), t)

            self.fim_do_quadro(t, dt)
            self._emitir(t)
            self.tracks = [tr for tr in self.tracks if tr.age <= m.max_age]

        return self._montar(name, fps, width, height, len(detections))

    # ------------------------------------------------------------------------ internos

    def _criar(self, box: np.ndarray, score: float, quadro: int) -> Track:
        track = Track(
            id=self._proximo_id, box=np.asarray(box, dtype=np.float32).copy(),
            score=score, nasceu_em=quadro, visto_em=quadro,
        )
        self._proximo_id += 1
        self.nascer(track)
        self.tracks.append(track)
        return track

    def _emitir(self, quadro: int) -> None:
        """Decide o que deste quadro entra na saída, respeitando ``min_hits``."""
        m = self.manager
        for track in self.tracks:
            observada_agora = track.visto_em == quadro
            if not observada_agora and not m.emitir_sem_observacao:
                continue

            if track.hits >= m.min_hits:
                # confirmada: despeja o que estava guardado nos quadros de origem e segue
                for t_antigo, box_antiga in track.pendentes:
                    self._saida[t_antigo].append((track.id, box_antiga))
                track.pendentes.clear()
                self._saida[quadro].append((track.id, track.box.copy()))
            else:
                track.pendentes.append((quadro, track.box.copy()))

    def _montar(
        self, name: str, fps: float, width: int, height: int, n_frames: int
    ) -> Sequence:
        frames = []
        for t in range(n_frames):
            registros = self._saida.get(t, [])
            ids = np.array([r[0] for r in registros], dtype=np.int64)
            boxes = np.array([r[1] for r in registros], dtype=np.float32).reshape(-1, 4)
            frames.append(Frame(
                index=t, boxes=boxes, ids=ids,
                visibility=np.ones(len(ids), dtype=np.float32),
            ))
        return Sequence(name=name, fps=fps, width=width, height=height, frames=frames)
