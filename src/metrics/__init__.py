"""Métricas do PA2.

``detection.py`` (AP das detecções) fica **fora** deste re-export de propósito: ele pertence
à Parte 1, e este pacote é o artefato 3 da Parte 0. Quem precisa da AP importa o módulo
direto — é o que ``src/baseline/runner.py`` e ``src/evaluation/tracking_engine.py`` fazem.
Manter a fronteira aqui é o que permite commitar a Parte 0 sozinha.
"""

from src.metrics.identity import (
    LIMIAR_IOU, IDF1Result, TrackingMetrics, compute_clearmot, compute_idf1, evaluate,
)
from src.metrics.iou import iou_matrix
from src.metrics.matching import match_clearmot, match_greedy, match_hungarian

__all__ = [
    "iou_matrix",
    "match_greedy", "match_hungarian", "match_clearmot",
    "LIMIAR_IOU", "IDF1Result", "TrackingMetrics",
    "compute_idf1", "compute_clearmot", "evaluate",
]
