"""O rastreador ingênuo da Parte 1 — associação por IoU entre quadros vizinhos.

É literalmente o que o enunciado descreve no item 2 da Parte 1:

    "IoU entre as detecções do quadro t e do quadro t−1, matching guloso ou Hungarian,
    liminar fixo, ID novo quando nada casa, track morta depois de k quadros sem observação."

O ponto deste rastreador não é ser bom: é **fracassar de um jeito mensurável**, para que o
gráfico do descolamento da Parte 1 tenha de onde sair. Ele não tem modelo de movimento
nenhum — a previsão para o quadro seguinte é a caixa onde o objeto estava, parada. Sob
oclusão o estado congela e envelhece até morrer.

Há dois modos, e a diferença entre eles é o argumento da Parte 2:

``velocidade=False`` (padrão)
    posição constante. É o baseline do enunciado, e o piso.

``velocidade=True``
    extrapolação linear a partir das duas últimas observações. Não é filtro de Kalman — é
    uma subtração —, mas é o **baseline honesto**: um modelo de movimento que não aprendeu
    nada. Ganhar da posição constante é fácil e não prova nada sobre recorrência; ganhar
    disto aqui é o que a Parte 2 precisa mostrar.
"""

import numpy as np

from src.tracking.base import Track, Tracker


class IoUTracker(Tracker):
    """Associação por IoU contra a última caixa conhecida de cada track.

    Args:
        velocidade: se a previsão extrapola linearmente a partir das duas últimas
            observações, em vez de repetir a última caixa.
        **kwargs: repassados ao ``TrackManager`` (``iou_threshold``, ``max_age``,
            ``min_hits``, ``min_score``, ``associacao``, ``emitir_sem_observacao``).
    """

    def __init__(self, velocidade: bool = False, **kwargs):
        self.velocidade = velocidade
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        #: id da track → (caixa observada antes da última, **em que quadro**). O quadro é
        #: indispensável: sem ele a diferença finita confunde "andou 125 px num quadro" com
        #: "andou 125 px em cinco". ``None`` enquanto a track só tiver uma observação.
        self._penultima: dict[int, tuple[np.ndarray, int] | None] = {}

    def nascer(self, track: Track) -> None:
        self._penultima[track.id] = None

    def prever(self, quadro: int) -> np.ndarray:
        """Onde cada track deve estar neste quadro.

        Sem velocidade, é a última caixa — o objeto "está onde estava". Com velocidade, a
        última caixa mais o deslocamento **por quadro**, multiplicado pelo tempo que passou
        desde a última observação (``age + 1``), de modo que a extrapolação continue durante
        a oclusão em vez de congelar.

        Note:
            O deslocamento por quadro é a diferença entre as duas últimas observações
            **dividida pelo número de quadros entre elas**. Dividir parece supérfluo — as
            observações costumam ser consecutivas — e não é: logo depois de uma oclusão, as
            duas últimas observações estão separadas por todo o buraco.

            Sem a divisão, uma track que atravessou 5 quadros escondida sai de lá achando
            que anda 5x mais rápido do que anda, e erra o quadro seguinte por 100 px. Foi
            exatamente o que aconteceu aqui: a track sobrevivia à oclusão e se perdia no
            quadro **seguinte** ao reencontro, o que parecia um problema de ``max_age``.

            É a mesma confusão entre "um quadro" e "um intervalo de tempo" que a Parte 5 vai
            explorar de propósito, e é por isso que o modelo da Parte 2 recebe Δt como
            entrada em vez de supor que ele vale sempre 1.
        """
        if not self.tracks:
            return np.empty((0, 4), dtype=np.float32)

        if not self.velocidade:
            return np.stack([t.box for t in self.tracks])

        previstas = []
        for track in self.tracks:
            anterior = self._penultima.get(track.id)
            if anterior is None:
                previstas.append(track.box)
            else:
                caixa_antiga, quadro_antigo = anterior
                intervalo = max(1, track.visto_em - quadro_antigo)
                por_quadro = (track.box - caixa_antiga) / intervalo
                previstas.append(track.box + por_quadro * (track.age + 1))
        return np.stack(previstas).astype(np.float32)

    def observar(self, track: Track, box: np.ndarray, score: float, quadro: int) -> None:
        self._penultima[track.id] = (track.box.copy(), track.visto_em)
        track.box = np.asarray(box, dtype=np.float32).copy()
        track.score = score
        track.visto_em = quadro
        track.hits += 1
        track.age = 0

    def sem_observacao(self, track: Track, quadro: int) -> None:
        """Sem detecção, o rastreador ingênuo não tem o que fazer além de envelhecer.

        A caixa fica **onde estava**: sem modelo de movimento, qualquer outra escolha seria
        inventada. É esta a lacuna que a Parte 2 preenche — lá a caixa continua andando
        porque o estado recorrente sabe para onde o objeto ia.

        Com ``velocidade=True`` a caixa também não se move aqui; quem extrapola é ``prever``,
        que é chamado a cada quadro e já leva ``age`` em conta. Deixar ``track.box`` parada
        mantém a última **observação** separada da última **previsão** — misturar as duas
        faria o erro de extrapolação se acumular sobre si mesmo a cada quadro de oclusão.
        """
        track.age += 1

    def config(self) -> dict:
        return {"tracker": "iou", "velocidade": self.velocidade, **self.manager.config()}
