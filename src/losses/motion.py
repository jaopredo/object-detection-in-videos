"""A perda da Parte 2, trilha A — ``smooth-L1`` sobre a caixa prevista.

O enunciado pergunta, sobre o PA1, *"qual perda otimiza isso (de onde vem o gradiente?)"*.
A resposta aqui é direta: o gradiente vem da distância entre a caixa que a recorrência
previu para o quadro seguinte e a caixa que o objeto de fato ocupa lá, medida em unidades da
própria caixa.

## Por que o resíduo, e não o incremento

O jeito óbvio seria comparar o incremento previsto com o incremento verdadeiro. Não serve:
sob oclusão simulada, a crença do modelo **é a previsão anterior dele**, então o incremento
verdadeiro dependeria de onde o modelo acha que está — e o gradiente teria um caminho para
tornar o alvo mais fácil em vez de a previsão melhor.

A perda aqui é sobre o **resíduo** ``delta(caixa_verdadeira, caixa_prevista)``, que é zero
quando as duas coincidem. O denominador é o tamanho da caixa **verdadeira**, que não depende
do modelo; o gradiente entra só pela previsão. E como a previsão de um passo vira a crença
do passo seguinte, ele atravessa o rollout inteiro — que é o BPTT cuja norma a Parte 4 mede.

## Por que smooth-L1 e não L2

Detecção e anotação têm caixas erradas — no MOT17, caixa amodal de pedestre atrás de carro é
um palpite do anotador. Com L2, um punhado dessas domina o gradiente da época. O smooth-L1 é
quadrático perto de zero (onde se quer precisão) e linear longe (onde provavelmente é
outlier), que é exatamente o comportamento desejado.
"""

import torch
import torch.nn as nn

from src.losses.factory import LossFactory, LossFactoryRegistry
from src.models.motion_rnn import delta_torch


class SmoothL1BoxLoss:
    """Distância entre a caixa prevista e a verdadeira, livre de escala.

    Args:
        beta: onde o smooth-L1 troca de quadrático para linear. Em unidades de caixa: 0,1
            é 10% da largura do objeto.
        peso_sob_oclusao: multiplicador da perda nos passos em que o modelo estava rodando
            às cegas. Maior que 1 concentra o gradiente justamente onde a recorrência tem
            que fazer o trabalho — atravessar o buraco — em vez de nos passos fáceis, em que
            copiar a última observação já quase resolve.
    """

    def __init__(self, beta: float = 0.1, peso_sob_oclusao: float = 1.0):
        self.beta = beta
        self.peso_sob_oclusao = peso_sob_oclusao
        self.fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def build_inputs(self, lote, device):
        return lote.to(device)

    def build_targets(self, lote, device):
        return lote.to(device)

    def __call__(self, saida: dict, alvo) -> tuple[torch.Tensor, dict]:
        """Perda do lote.

        O passo ``t`` prevê a caixa de ``t+1``, então o último passo da janela não tem alvo e
        fica de fora. Uma janela de T passos contribui com T−1 termos.
        """
        previsao = saida["previsao"][:, :-1]        # (B, T-1, 4)
        verdadeira = alvo.caixas[:, 1:]             # (B, T-1, 4)

        residuo = delta_torch(verdadeira, previsao)
        por_coordenada = self.fn(residuo, torch.zeros_like(residuo))
        por_passo = por_coordenada.mean(dim=-1)     # (B, T-1)

        # o passo t+1 é "às cegas" quando não houve observação nele
        cego = 1.0 - alvo.observado[:, 1:]
        peso = 1.0 + (self.peso_sob_oclusao - 1.0) * cego

        total = (por_passo * peso).sum() / peso.sum().clamp(min=1.0)

        with torch.no_grad():
            n_cego = cego.sum().clamp(min=1.0)
            n_visto = (1 - cego).sum().clamp(min=1.0)
            componentes = {
                "visto": float((por_passo * (1 - cego)).sum() / n_visto),
                "cego": float((por_passo * cego).sum() / n_cego),
            }
        return total, componentes


class SmoothL1BoxLossFactory(LossFactory):
    def build(self, cfg):
        perda = (cfg.model.get("loss") or {})
        return SmoothL1BoxLoss(
            beta=perda.get("beta", 0.1),
            peso_sob_oclusao=perda.get("peso_sob_oclusao", 1.0),
        )


LossFactoryRegistry._factories["smooth_l1_box"] = SmoothL1BoxLossFactory()
