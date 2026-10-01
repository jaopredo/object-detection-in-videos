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

    @staticmethod
    def _carregar_registros() -> None:
        """Importa o pacote, o que executa os ``_factories[...] = ...`` dos módulos.

        Sem isto o registry fica vazio quando alguém importa só a factory — que é o que o
        ``TrainEngine`` faz. O import fica aqui dentro, e não no topo, porque o módulo da
        factory é importado *pelos* modelos: no topo seria circular.
        """
        import importlib
        importlib.import_module("src.models")

    @classmethod
    def get(cls, cfg) -> ModelFactory:
        nome = cfg.model.get("name")
        cls._carregar_registros()
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
