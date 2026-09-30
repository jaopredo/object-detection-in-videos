"""IDF1, ID switches e fragmentações — implementação nossa.

O enunciado nomeia este arquivo como entregável e proíbe ``motmetrics``, ``TrackEval`` e
``py-motmetrics``. A implementação está em ``src/metrics/``, dividida em três peças com
responsabilidades separadas:

    src/metrics/iou.py        IoU entre conjuntos de caixas, vetorizada
    src/metrics/matching.py   as três regras de casamento (guloso, Hungarian, CLEAR-MOT)
    src/metrics/identity.py   IDF1 global, ID switches, fragmentações, MOTA

Este módulo existe para que ``metrics.py`` seja um arquivo de verdade, como o enunciado
pede, e para dar um ponto de entrada curto::

    from metrics import evaluate
    m = evaluate(gabarito, predicao)
    print(m.idf1, m.id_switches, m.fragmentations)

Os testes que provam que as contas estão certas estão em ``tests/test_metrics.py`` — os três
casos construídos à mão que o enunciado exige, mais os degenerados.
"""

from src.metrics.identity import (
    LIMIAR_IOU,
    IDF1Result,
    TrackingMetrics,
    compute_clearmot,
    compute_idf1,
    evaluate,
)
from src.metrics.iou import iou_matrix
from src.metrics.matching import match_clearmot, match_greedy, match_hungarian

__all__ = [
    "evaluate", "compute_idf1", "compute_clearmot",
    "TrackingMetrics", "IDF1Result", "LIMIAR_IOU",
    "iou_matrix", "match_greedy", "match_hungarian", "match_clearmot",
]
