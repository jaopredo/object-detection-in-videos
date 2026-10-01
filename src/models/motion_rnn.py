"""O modelo temporal da Parte 2, trilha A — uma RNN como modelo de movimento.

Um estado recorrente **por track**. A cada quadro ele recebe a última observação (caixa,
confiança, Δt) e prevê a caixa do quadro seguinte; a associação usa IoU entre a caixa
prevista e a observada. Sob oclusão o estado roda para a frente sem observação.

## A decisão que faz esta parte funcionar

``forward`` **é** o laço do rastreador. Não é parecido: é o mesmo.

A cada passo o modelo decide, pela máscara de observação, se atualiza a crença com a caixa
que chegou ou com a própria previsão anterior — que é exatamente o que ``RNNTracker`` faz
quadro a quadro. Treinar com um laço e inferir com outro é como se cria o descolamento de
distribuição do Eixo 2 do enunciado: *"na inferência o modelo se alimenta das próprias
previsões (e, sob oclusão, só delas); se ele nunca viu isso no treino, a distribuição muda
debaixo dele"*.

Com o laço compartilhado, ligar oclusão simulada no treino (mascarar passos ao acaso) põe o
modelo exatamente no regime de inferência, e não há uma segunda implementação para
divergir da primeira.

## O que entra em cada passo (11 números)

===============  ====  ==========================================================
``tx ty tw th``     4  o movimento **observado** ao chegar neste passo
``cx cy w h``       4  onde a caixa está no quadro, dividido pelo tamanho do quadro
``dt``              1  segundos desde a observação anterior — o botão da Parte 5
``observado``       1  1 se chegou detecção, 0 se está rodando às cegas
``score``           1  confiança da detecção (0 quando não houve)
===============  ====  ==========================================================

``dt`` em **segundos**, não em quadros. O MOT17 tem sequências a 14, 25 e 30 fps; em quadros,
"um passo" significa três coisas diferentes no mesmo dataset. Em segundos, o mesmo pedestre
andando na mesma velocidade gera o mesmo alvo nas três — e a Parte 5, que subamostra o vídeo,
vira uma mudança de uma entrada que o modelo já conhece em vez de um regime que ele nunca viu.
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.factory import ModelFactory, ModelFactoryRegistry

#: tx ty tw th · cx cy w h · dt · observado · score
DIM_ENTRADA = 11

#: mesmo piso de ``src/boxes.py``: caixa de dimensão zero daria divisão por zero e log(0).
EPS = 1e-3

#: limite do expoente ao aplicar a mudança de escala. Sem ele, um ``tw = 20`` no começo do
#: treino vira uma caixa de 10^8 px e o NaN se espalha pelo lote inteiro.
LIMITE_LOG = 4.0

CELULAS = {"rnn": nn.RNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}


def delta_torch(de: torch.Tensor, para: torch.Tensor) -> torch.Tensor:
    """Incremento entre duas caixas ``cxcywh``, na parametrização de ``src/boxes.py``."""
    w = de[..., 2].clamp(min=EPS)
    h = de[..., 3].clamp(min=EPS)
    return torch.stack([
        (para[..., 0] - de[..., 0]) / w,
        (para[..., 1] - de[..., 1]) / h,
        torch.log(para[..., 2].clamp(min=EPS) / w),
        torch.log(para[..., 3].clamp(min=EPS) / h),
    ], dim=-1)


def aplicar_delta_torch(caixa: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Aplica o incremento e devolve a caixa nova, em ``cxcywh``."""
    w = caixa[..., 2].clamp(min=EPS)
    h = caixa[..., 3].clamp(min=EPS)
    return torch.stack([
        caixa[..., 0] + d[..., 0] * w,
        caixa[..., 1] + d[..., 1] * h,
        w * torch.exp(d[..., 2].clamp(-LIMITE_LOG, LIMITE_LOG)),
        h * torch.exp(d[..., 3].clamp(-LIMITE_LOG, LIMITE_LOG)),
    ], dim=-1)


class MotionRNN(nn.Module):
    """Estado recorrente por track, prevendo o incremento da caixa.

    Args:
        cell: ``"rnn"``, ``"lstm"`` ou ``"gru"``. É o Eixo 1 da ablação da Parte 3.
        hidden: tamanho do estado oculto.
        layers: quantas células empilhadas.
        prever_incerteza: acrescenta uma cabeça de log-variância por coordenada, para a
            NLL gaussiana e o portão de associação adaptativo. Opcional no enunciado.
    """

    def __init__(
        self,
        cell: str = "gru",
        hidden: int = 64,
        layers: int = 1,
        prever_incerteza: bool = False,
    ):
        super().__init__()
        if cell not in CELULAS:
            raise ValueError(f"cell {cell!r} não existe. Use um de {sorted(CELULAS)}.")

        self.nome_celula = cell
        self.hidden = hidden
        self.layers = layers
        self.prever_incerteza = prever_incerteza
        self.tem_estado_de_celula = cell == "lstm"   # LSTM carrega (h, c); as outras só h

        self.entrada = nn.Sequential(
            nn.Linear(DIM_ENTRADA, hidden), nn.ReLU(),
        )
        self.celulas = nn.ModuleList(
            [CELULAS[cell](hidden, hidden) for _ in range(layers)]
        )
        self.cabeca = nn.Linear(hidden, 8 if prever_incerteza else 4)

        # a cabeça começa prevendo incremento zero — "o objeto continua onde está". É o
        # palpite certo na média (a maioria dos passos é de movimento pequeno), então o
        # treino começa perto de uma solução razoável em vez de num chute que produz caixas
        # gigantes e satura o clamp do expoente.
        nn.init.zeros_(self.cabeca.weight)
        nn.init.zeros_(self.cabeca.bias)

    # ------------------------------------------------------------------- estado oculto

    def estado_inicial(self, n: int, device) -> list:
        zeros = lambda: torch.zeros(n, self.hidden, device=device)
        if self.tem_estado_de_celula:
            return [(zeros(), zeros()) for _ in range(self.layers)]
        return [zeros() for _ in range(self.layers)]

    def passo(self, x: torch.Tensor, estado: list) -> tuple[torch.Tensor, list]:
        """Um passo da recorrência. Devolve (incremento previsto, estado novo).

        Público porque o ``RNNTracker`` chama isto quadro a quadro, com uma linha por track
        viva. É o mesmo caminho de código que o ``forward`` percorre no treino.
        """
        z = self.entrada(x)
        novo = []
        for celula, anterior in zip(self.celulas, estado):
            saida = celula(z, anterior)
            novo.append(saida)
            z = saida[0] if self.tem_estado_de_celula else saida
        return self.cabeca(z), novo

    # ---------------------------------------------------------------------- entrada

    @staticmethod
    def montar_entrada(
        d_observado: torch.Tensor, caixa: torch.Tensor, tamanho: torch.Tensor,
        dt: torch.Tensor, observado: torch.Tensor, score: torch.Tensor,
    ) -> torch.Tensor:
        """Os 11 números de um passo, a partir da crença atual e do que chegou."""
        contexto = torch.stack([
            caixa[..., 0] / tamanho[..., 0], caixa[..., 1] / tamanho[..., 1],
            caixa[..., 2] / tamanho[..., 0], caixa[..., 3] / tamanho[..., 1],
        ], dim=-1)
        return torch.cat([
            d_observado, contexto,
            dt.unsqueeze(-1), observado.unsqueeze(-1), score.unsqueeze(-1),
        ], dim=-1)

    # ---------------------------------------------------------------------- rollout

    def forward(self, lote) -> dict:
        """Roda um ``JanelaLote``. É o que o ``TrainEngine`` chama: ``model(entrada)``."""
        return self.rollout(
            lote.caixas, lote.observado, lote.dt, lote.score, lote.tamanho
        )

    def rollout(
        self,
        caixas: torch.Tensor,     # (B, T, 4) cxcywh em pixels — o gabarito da janela
        observado: torch.Tensor,  # (B, T) 1 se a caixa deste passo foi "vista"
        dt: torch.Tensor,         # (B, T) segundos desde o passo anterior
        score: torch.Tensor,      # (B, T)
        tamanho: torch.Tensor,    # (B, 2) largura e altura do quadro
        reter_estados: bool = False,
    ) -> dict:
        """Roda a janela inteira, realimentando a própria previsão nos passos não observados.

        Returns:
            ``crenca``  (B, T, 4) onde o modelo **acreditava** estar em cada passo. Nos
                        passos observados é a caixa que chegou; nos demais é a previsão que
                        ele mesmo fez. É a partir dela que o alvo é montado — o modelo tem
                        que aprender a ir *de onde acha que está* para onde o objeto está,
                        que é o problema que ele de fato enfrenta sob oclusão.
            ``incremento``  (B, T, 4) o que a cabeça previu em cada passo.
            ``previsao``    (B, T, 4) a caixa prevista para o passo **seguinte**.
            ``log_var``     (B, T, 4) ou ``None``.
            ``estados``     lista de T tensores (B, hidden) — só com ``reter_estados``.
                            É o que a Parte 4 deriva para medir ‖∂L_t/∂h_{t−k}‖: sem
                            guardar os estados intermediários não há em relação a que
                            derivar. Fica desligado por padrão porque no treino eles são
                            lixo que só ocupa memória.
        """
        B, T, _ = caixas.shape
        tamanho = tamanho.unsqueeze(1) if tamanho.dim() == 2 else tamanho

        estado = self.estado_inicial(B, caixas.device)
        crenca = caixas[:, 0]
        anterior = caixas[:, 0]

        saida_crenca, saida_incremento, saida_previsao, saida_logvar = [], [], [], []
        saida_estados = []

        for t in range(T):
            d_observado = delta_torch(anterior, crenca)
            x = self.montar_entrada(
                d_observado, crenca, tamanho[:, 0], dt[:, t], observado[:, t], score[:, t]
            )
            bruto, estado = self.passo(x, estado)
            if reter_estados:
                # a LSTM carrega (h, c); a norma do gradiente é medida no h, que é o que
                # atravessa para o passo seguinte como saída da célula
                oculto = estado[-1][0] if self.tem_estado_de_celula else estado[-1]
                oculto.retain_grad()
                saida_estados.append(oculto)
            incremento = bruto[:, :4]
            log_var = bruto[:, 4:] if self.prever_incerteza else None

            previsao = aplicar_delta_torch(crenca, incremento)

            saida_crenca.append(crenca)
            saida_incremento.append(incremento)
            saida_previsao.append(previsao)
            if log_var is not None:
                saida_logvar.append(log_var)

            if t + 1 < T:
                anterior = crenca
                visto = observado[:, t + 1].unsqueeze(-1)
                # onde chegou detecção, a crença vira a observação; onde não chegou, vira a
                # previsão. É literalmente o passo 3/4 do RNNTracker.
                crenca = visto * caixas[:, t + 1] + (1 - visto) * previsao

        empilha = lambda lista: torch.stack(lista, dim=1)
        return {
            "crenca": empilha(saida_crenca),
            "incremento": empilha(saida_incremento),
            "previsao": empilha(saida_previsao),
            "log_var": empilha(saida_logvar) if saida_logvar else None,
            "estados": saida_estados if reter_estados else None,
        }

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {"cell": self.nome_celula, "hidden": self.hidden, "layers": self.layers,
                "prever_incerteza": self.prever_incerteza}


def hidden_para_orcamento(cell: str, alvo: int, layers: int = 1,
                          prever_incerteza: bool = False) -> int:
    """Maior ``hidden`` cujo modelo não passa de ``alvo`` parâmetros.

    O Eixo 1 da Parte 3 compara RNN simples, LSTM e GRU "no mesmo orçamento aproximado de
    parâmetros". A conta à mão é fácil de errar — a RNN tem 1 porta, a GRU 3 e a LSTM 4, e
    ainda há a camada de entrada e a cabeça —, então o orçamento é **procurado** e depois
    conferido, em vez de calculado e confiado.
    """
    melhor = 1
    for h in range(1, 4096):
        n = MotionRNN(cell, h, layers, prever_incerteza).n_parametros()
        if n > alvo:
            break
        melhor = h
    return melhor


class MotionRNNFactory(ModelFactory):
    """Constrói o ``MotionRNN`` a partir do bloco ``model`` do YAML."""

    def build(self, cfg):
        m = cfg.model
        hidden = m.get("hidden")
        if hidden is None:
            # sem `hidden` explícito, o config declara um orçamento e a célula se ajusta a
            # ele — é o que mantém a comparação do Eixo 1 honesta sem 12 YAMLs
            hidden = hidden_para_orcamento(
                m.get("cell", "gru"), m.get("param_budget", 20000),
                m.get("layers", 1), m.get("prever_incerteza", False),
            )
        return MotionRNN(
            cell=m.get("cell", "gru"), hidden=int(hidden),
            layers=m.get("layers", 1),
            prever_incerteza=m.get("prever_incerteza", False),
        )


ModelFactoryRegistry._factories["motion_rnn"] = MotionRNNFactory()
