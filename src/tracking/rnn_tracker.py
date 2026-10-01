"""O rastreador da Parte 2 — um estado recorrente por track.

Responde as três perguntas que o enunciado diz que o PA1 não entregou:

**O que a recorrência carrega.** O estado oculto de uma célula por track, alimentado com a
caixa observada, o Δt e a confiança. Ele não é a caixa: é o resumo de como aquele objeto
vinha se movendo, e é o que sobrevive quando a observação some.

**De onde vem o gradiente.** Da distância entre a caixa prevista e a verdadeira do quadro
seguinte, retropropagada pelo rollout inteiro — ver ``src/losses/motion.py``.

**Como decodificar em trajetórias.** O laço abaixo. Por quadro:

1. cada track viva já trouxe do quadro anterior a sua **caixa prevista**;
2. IoU entre as previstas e as detecções → Hungarian, com o mesmo limiar da Parte 1;
3. casou → a crença vira a observação;
4. não casou → **a crença vira a própria previsão**, e ``age`` sobe. É aqui que a oclusão
   é atravessada, e é a única diferença real em relação ao rastreador ingênuo, cuja crença
   fica parada;
5. detecção sem par → nasce track com estado zerado;
6. ``age > max_age`` → morte.

A gestão de nascimento e morte é **a mesma** do ``IoUTracker``, herdada de ``TrackManager``.
Isso é deliberado: se cada um gerisse tracks do seu jeito, a comparação entre os dois
mediria as duas coisas ao mesmo tempo e não diria nada sobre a recorrência.
"""

import numpy as np
import torch

from src.boxes import cxcywh_para_xyxy, xyxy_para_cxcywh
from src.models.motion_rnn import MotionRNN, aplicar_delta_torch, delta_torch
from src.tracking.base import Track, Tracker


class RNNTracker(Tracker):
    """Associação por IoU contra a caixa que a recorrência prevê.

    Args:
        model: o ``MotionRNN`` treinado. Usado em modo de inferência; nunca recebe gradiente.
        device: onde rodar. CPU é o bastante — o modelo tem ~20k parâmetros.
        **kwargs: repassados ao ``TrackManager``. O padrão liga ``emitir_sem_observacao``,
            porque sustentar a caixa durante a oclusão é o ponto deste rastreador.
    """

    def __init__(self, model: MotionRNN, device: str = "cpu", **kwargs):
        kwargs.setdefault("emitir_sem_observacao", True)
        self.model = model.to(device).eval()
        self.device = device
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        #: id da track → estado oculto (lista de tensores, um por camada)
        self._estado: dict[int, list] = {}
        #: id → (crença anterior, crença atual, previsão para o próximo quadro), em cxcywh
        self._crenca: dict[int, np.ndarray] = {}
        self._anterior: dict[int, np.ndarray] = {}
        self._previsao: dict[int, np.ndarray] = {}
        #: se a track recebeu observação **neste** quadro, e com que confiança
        self._observado: dict[int, float] = {}
        self._score: dict[int, float] = {}

    # ------------------------------------------------------------------------ o contrato

    def nascer(self, track: Track) -> None:
        """Estado zerado e crença igual à detecção que a criou."""
        caixa = xyxy_para_cxcywh(track.box)[0]
        self._estado[track.id] = self.model.estado_inicial(1, self.device)
        self._crenca[track.id] = caixa
        self._anterior[track.id] = caixa.copy()   # sem passado: deslocamento observado zero
        self._previsao[track.id] = caixa.copy()
        self._observado[track.id] = 1.0
        self._score[track.id] = track.score

    def prever(self, quadro: int) -> np.ndarray:
        """As caixas que a recorrência previu para este quadro, no quadro anterior."""
        if not self.tracks:
            return np.empty((0, 4), dtype=np.float32)
        return cxcywh_para_xyxy(
            np.stack([self._previsao[t.id] for t in self.tracks])
        ).astype(np.float32)

    def observar(self, track: Track, box: np.ndarray, score: float, quadro: int) -> None:
        """Chegou detecção: a crença vira a observação."""
        self._anterior[track.id] = self._crenca[track.id]
        self._crenca[track.id] = xyxy_para_cxcywh(box)[0]
        self._observado[track.id] = 1.0
        self._score[track.id] = score

        track.box = np.asarray(box, dtype=np.float32).copy()
        track.score = score
        track.visto_em = quadro
        track.hits += 1
        track.age = 0

    def sem_observacao(self, track: Track, quadro: int) -> None:
        """Não chegou detecção: a crença vira a **própria previsão**.

        É a única linha que separa este rastreador do ingênuo. Lá a caixa fica parada porque
        não há modelo de movimento; aqui ela continua andando, porque o estado sabe para
        onde o objeto ia.
        """
        self._anterior[track.id] = self._crenca[track.id]
        self._crenca[track.id] = self._previsao[track.id]
        self._observado[track.id] = 0.0
        self._score[track.id] = 0.0

        track.box = cxcywh_para_xyxy(self._crenca[track.id])[0].astype(np.float32)
        track.age += 1

    @torch.no_grad()
    def fim_do_quadro(self, quadro: int, dt: float) -> None:
        """Avança a recorrência de todas as tracks vivas, numa chamada só.

        É o mesmo passo de ``MotionRNN.rollout``, com uma linha por track em vez de uma por
        janela. Batelar importa: um quadro do MOT17-04 tem ~60 tracks vivas, e sessenta
        chamadas de célula com uma linha cada custam muito mais que uma com sessenta.
        """
        if not self.tracks:
            return

        ids = [t.id for t in self.tracks]
        tensor = lambda arr: torch.as_tensor(np.stack(arr), dtype=torch.float32,
                                             device=self.device)

        crenca = tensor([self._crenca[i] for i in ids])
        anterior = tensor([self._anterior[i] for i in ids])
        tamanho = torch.tensor([[self.width, self.height]], dtype=torch.float32,
                               device=self.device).expand(len(ids), 2)

        x = self.model.montar_entrada(
            delta_torch(anterior, crenca), crenca, tamanho,
            torch.full((len(ids),), dt, device=self.device),
            torch.tensor([self._observado[i] for i in ids], device=self.device),
            torch.tensor([self._score[i] for i in ids], device=self.device),
        )

        # o estado oculto é guardado por track; aqui as linhas são empilhadas na ordem de
        # `ids`, passam pela célula juntas e voltam para os seus donos
        estado = self._empilhar([self._estado[i] for i in ids])
        bruto, estado = self.model.passo(x, estado)
        for k, i in enumerate(ids):
            self._estado[i] = self._fatiar(estado, k)

        previsao = aplicar_delta_torch(crenca, bruto[:, :4]).cpu().numpy()
        for k, i in enumerate(ids):
            self._previsao[i] = previsao[k]

    # ------------------------------------------------------- empacotar o estado oculto

    def _empilhar(self, estados: list[list]) -> list:
        """Junta os estados de várias tracks num lote, camada por camada."""
        if self.model.tem_estado_de_celula:   # LSTM: (h, c) por camada
            return [
                (torch.cat([e[camada][0] for e in estados]),
                 torch.cat([e[camada][1] for e in estados]))
                for camada in range(self.model.layers)
            ]
        return [torch.cat([e[camada] for e in estados])
                for camada in range(self.model.layers)]

    def _fatiar(self, estado: list, k: int) -> list:
        """A linha ``k`` do estado em lote, de volta ao formato de uma track."""
        if self.model.tem_estado_de_celula:
            return [(c[0][k:k + 1], c[1][k:k + 1]) for c in estado]
        return [c[k:k + 1] for c in estado]

    def config(self) -> dict:
        return {"tracker": "rnn", **self.model.config(), **self.manager.config()}
