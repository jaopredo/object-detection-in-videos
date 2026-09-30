"""Constrói o modelo temporal a partir da configuração.

Mesmo padrão do PA1 (``semantic-segmentation/src/models/factory.py``): trocar ``model.name``
no YAML troca a arquitetura, e o registry recusa nomes que não existem.

**Ainda não há nada registrado.** A Parte 0 é só geração de dados e a Parte 1 roda com a
fonte de detecções congelada e associação por IoU — nenhuma das duas treina nada. As
entradas (``rnn``, ``lstm``, ``gru``) chegam na Parte 2, junto com o modelo temporal.

O registry vazio existe desde já de propósito: ``--mode train`` falha com uma mensagem que
diz o que está faltando, em vez de quebrar num ``ImportError`` dez frames acima.
"""

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    """Contrato comum das factories de modelo."""

    @abstractmethod
    def build(self, cfg):
        """Instancia o modelo a partir de um TrainConfig ou EvalConfig."""


class ModelFactoryRegistry:
    _factories: dict[str, ModelFactory] = {}

    @classmethod
    def get(cls, cfg) -> ModelFactory:
        nome = cfg.model.get("name")
        if not cls._factories:
            raise NotImplementedError(
                "nenhum modelo temporal registrado ainda — o modelo da Parte 2 (trilha A "
                "ou B) entra em src/models/. Até lá, as partes 0 e 1 rodam sem treino: "
                "use --mode gen-synth."
            )
        if nome not in cls._factories:
            raise ValueError(
                f"model.name desconhecido: {nome!r}. "
                f"Disponíveis: {sorted(cls._factories)}"
            )
        return cls._factories[nome]

    @classmethod
    def build(cls, cfg):
        """Atalho: escolhe a factory e já instancia."""
        return cls.get(cfg).build(cfg)
