"""Modelos temporais do PA2.

Importar o pacote registra as factories em ``ModelFactoryRegistry`` — é o que faz
``model.name: motion_rnn`` no YAML encontrar alguma coisa.
"""

from src.models.motion_rnn import MotionRNN, hidden_para_orcamento

__all__ = ["MotionRNN", "hidden_para_orcamento"]
