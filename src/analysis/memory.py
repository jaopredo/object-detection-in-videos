"""Parte 4 — o horizonte de memória do modelo, medido das duas formas obrigatórias.

O enunciado exige: *"meçam o horizonte de memória efetivo do seu modelo, das duas formas"* —
analítica (a norma do gradiente) e empírica (quantos quadros o estado sobrevive a uma
oclusão). E manda comparar a curva empírica com a **distribuição de duração de oclusão do
dataset**, que é a régua contra a qual o número só significa alguma coisa.

As duas medidas respondem perguntas diferentes e têm que contar a mesma história:

**analítica** — até que distância no passado o treino conseguiu **mandar sinal**. Se
``‖∂L_t/∂h_{t−k}‖`` já é desprezível em k = 8, nada que o modelo faça no quadro t foi
ensinado por causa do que aconteceu 8 quadros antes: o gradiente não chegou lá.

**empírica** — até que distância o estado **serve** na hora de rastrear. É o que se vê no
vídeo.

Se as duas discordarem, uma das duas medições está errada — e descobrir isso é resultado.
"""

import numpy as np
import torch

from src.evaluation.occlusion import occlusion_runs
from src.metrics.iou import iou_matrix
from src.metrics.matching import match_clearmot


# ============================================================== 1. medida analítica

def norma_do_gradiente_por_distancia(
    model, lote, passo_final: int | None = None
) -> np.ndarray:
    """``‖∂L_t/∂h_{t−k}‖`` em função de ``k``, com ``t`` no fim da janela.

    A perda é tomada **num passo só** — o último da janela — de propósito. Com a perda
    somada sobre a janela inteira, o gradiente em ``h_{t−k}`` misturaria o caminho longo
    (que é o que se quer medir) com os caminhos curtos vindos dos passos intermediários, e a
    curva ficaria plana por construção.

    Args:
        model: o ``MotionRNN``.
        lote: um ``JanelaLote``.
        passo_final: em que passo medir a perda. ``None`` usa o penúltimo, que é o último
            que tem alvo.

    Returns:
        (K,) norma média por amostra, indexada por ``k = 0, 1, ...`` — distância em passos
        para trás a partir de ``passo_final``.
    """
    from src.models.motion_rnn import delta_torch

    model.zero_grad(set_to_none=True)
    saida = model.rollout(lote.caixas, lote.observado, lote.dt, lote.score,
                          lote.tamanho, reter_estados=True)
    estados = saida["estados"]
    t = (len(estados) - 2) if passo_final is None else passo_final

    residuo = delta_torch(lote.caixas[:, t + 1], saida["previsao"][:, t])
    perda = residuo.abs().mean()

    gradientes = torch.autograd.grad(perda, estados[: t + 1], retain_graph=False,
                                     allow_unused=True)
    # normalizada por amostra: a norma de um lote de 256 é 16x a de um lote de 1 só por
    # causa do tamanho, e a comparação entre células ficaria refém do lote
    normas = [
        float(g.norm(dim=-1).mean()) if g is not None else 0.0 for g in gradientes
    ]
    return np.array(normas[::-1])   # índice 0 = o próprio passo t, 1 = um antes, ...


def horizonte_analitico(normas: np.ndarray, fracao: float = 0.01) -> int:
    """Até que ``k`` o gradiente ainda vale ``fracao`` do que vale em ``k = 0``.

    Um corte é arbitrário por natureza; 1% é o que se usa para "morreu". O número tem que
    ser lido junto da curva, nunca sozinho.
    """
    if not len(normas) or normas[0] <= 0:
        return 0
    relativa = normas / normas[0]
    acima = np.flatnonzero(relativa >= fracao)
    return int(acima[-1]) if len(acima) else 0


# =============================================================== 2. medida empírica

def correspondencia_por_quadro(gt, pred, threshold: float = 0.5) -> list[dict]:
    """Para cada quadro, o mapa identidade verdadeira → identidade prevista.

    Usa o mesmo casamento do ``compute_clearmot`` — o que preserva o par do quadro anterior
    —, de modo que "sobreviveu à oclusão" aqui signifique a mesma coisa que "não contou ID
    switch" lá. Duas definições diferentes de casamento dariam duas histórias diferentes
    sobre o mesmo vídeo.
    """
    ultimo_par: dict[int, int] = {}
    por_quadro = []
    for quadro_gt, quadro_pred in zip(gt, pred):
        iou = iou_matrix(quadro_gt.boxes, quadro_pred.boxes)
        pares = match_clearmot(iou, quadro_gt.ids, quadro_pred.ids, ultimo_par, threshold)
        mapa = {}
        for i, j in pares:
            g, p = int(quadro_gt.ids[i]), int(quadro_pred.ids[j])
            mapa[g] = p
            ultimo_par[g] = p
        por_quadro.append(mapa)
    return por_quadro


def sobrevivencia_a_oclusao(gt, pred, threshold: float = 0.5) -> list[dict]:
    """Para cada oclusão do gabarito, se a identidade prevista sobreviveu a ela.

    Uma oclusão é um trecho em que a identidade verdadeira existe, some e volta (ver
    ``src/evaluation/occlusion.py``). Sobreviver é sair do outro lado com **o mesmo
    identificador previsto** — que é literalmente o enunciado: *"se um objeto aparece no
    quadro 3 e reaparece no quadro 40, ele tem que sair com o mesmo identificador"*.

    Returns:
        Uma lista de ``{"duracao", "sobreviveu", "track_id", "antes", "depois"}``. Oclusões
        em que a identidade não estava sendo rastreada antes do buraco ficam de fora: não dá
        para perder o que não se tinha.
    """
    mapas = correspondencia_por_quadro(gt, pred, threshold)
    saida = []
    for run in occlusion_runs(gt):
        antes = mapas[run.last_seen].get(run.track_id)
        if antes is None:
            continue
        depois = mapas[run.next_seen].get(run.track_id)
        saida.append({
            "track_id": run.track_id, "duracao": run.duration,
            "antes": antes, "depois": depois,
            "sobreviveu": bool(depois is not None and depois == antes),
        })
    return saida


def horizonte_empirico(eventos: list[dict], minimo: float = 0.5,
                       n_minimo: int = 5) -> int:
    """Até quantos quadros de buraco mais da metade das identidades ainda volta.

    Lido como: "até N quadros, o rastreador costuma trazer a identidade de volta; além
    disso, costuma não trazer".

    Note:
        A varredura **para na primeira faixa que falha**, e exige ``n_minimo`` eventos para
        uma faixa contar. As duas condições existem por causa de um resultado errado:
        a primeira versão pegava a maior duração com taxa ≥ 50%, sem mínimo de amostra, e
        devolveu **36 quadros** num dataset em que o gradiente morre em 6 — porque havia uma
        oclusão de 36 quadros, uma só, que por acaso sobreviveu. Um evento não é uma taxa.

        Parar na primeira falha também importa: sobrevivência não é monótona com a duração
        (buracos longos de objeto isolado são mais fáceis que buracos curtos em multidão), e
        sem parar o estimador pula por cima do ponto em que o modelo de fato quebrou.
    """
    if not eventos:
        return 0

    horizonte = 0
    for faixa in taxa_por_duracao(eventos):
        if faixa["n"] < n_minimo:
            break
        if faixa["taxa"] < minimo:
            break
        horizonte = faixa["max"] if faixa["max"] < 10 ** 5 else faixa["min"]
    return horizonte


def taxa_por_duracao(eventos: list[dict], bins=(1, 3, 6, 11, 21, 41, 10**6)) -> list[dict]:
    """Taxa de sobrevivência agrupada por faixa de duração da oclusão."""
    saida = []
    for inicio, fim in zip(bins[:-1], bins[1:]):
        no_grupo = [e for e in eventos if inicio <= e["duracao"] < fim]
        if not no_grupo:
            continue
        saida.append({
            "min": inicio, "max": fim - 1, "n": len(no_grupo),
            "taxa": float(np.mean([e["sobreviveu"] for e in no_grupo])),
        })
    return saida
