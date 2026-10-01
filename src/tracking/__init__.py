"""Rastreamento: tracks, gestão de nascimento e morte, NMS, e as regras de associação.

``rnn_tracker`` fica **fora** deste re-export de propósito: ele é da Parte 2 e depende de
``src/models/``, enquanto este pacote é a Parte 1 e roda sem torch. Quem precisa do
rastreador recorrente importa o módulo direto — ``from src.tracking.rnn_tracker import
RNNTracker``. Manter a fronteira aqui é o que permite commitar a Parte 1 sozinha, e o que
deixa a Parte 1 inteira rodar sem o torch instalado.
"""

from src.tracking.base import Track, TrackManager, Tracker
from src.tracking.iou_tracker import IoUTracker
from src.tracking.nms import nms, nms_por_classe

__all__ = ["Track", "TrackManager", "Tracker", "IoUTracker", "nms", "nms_por_classe"]
