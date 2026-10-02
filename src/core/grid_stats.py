"""Média e desvio por métrica entre seeds — usado pela ablação e pelo grid search.

Separar o efeito do ruído de seed é a mesma régua nos dois casos: o PA1 a usou para validar
o Eixo 3, a ablação da Parte 3 para a célula recorrente, e o grid search de hiperparâmetros
para decidir ``lr``/``batch_size`` sem escolher por sorte de inicialização.
"""

import numpy as np


def mean_std_metrics(por_seed: dict[int, dict], metric_names: tuple[str, ...]) -> dict:
    """Para cada métrica em ``metric_names``, a média e o desvio entre as seeds de
    ``por_seed`` (``{seed: {metrica: valor, ...}}``), mais os valores crus por seed.

    Métricas ausentes em algum seed (ex.: ``ap`` removido do grid search) são ignoradas em
    vez de quebrar a agregação.
    """
    saida = {}
    for metrica in metric_names:
        valores = [v[metrica] for v in por_seed.values() if metrica in v]
        if valores:
            saida[metrica] = {
                "mean": float(np.mean(valores)), "std": float(np.std(valores)),
                "values": {str(s): v[metrica] for s, v in por_seed.items() if metrica in v},
            }
    return saida
