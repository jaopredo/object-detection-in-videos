"""Escolhe a perda a partir do config — mesmo padrão do ``ModelFactoryRegistry``.

Como no PA1, a perda não é responsabilidade do modelo: o YAML declara ``model.loss.name`` e
esta factory instancia o objeto de tarefa correspondente.

**Ainda não há nada registrado** — ver a mesma observação em ``src/models/factory.py``. As
entradas chegam na Parte 2:

    trilha A   L1/smooth-L1 sobre a caixa prevista, treinada em trajetórias do ground truth
    trilha B   contrastiva ou triplet sobre as identidades do ground truth
"""

from abc import ABC, abstractmethod


def resolve_task_name(cfg) -> str:
    """Nome da tarefa a partir de ``cfg.model``, sem levantar erro.

    Leitura pura — nunca muta ``cfg.model``, que é compartilhado entre o TrainConfig e o
    EvalConfig do mesmo YAML.
    """
    return (cfg.model.get("loss") or {}).get("name", "")


class LossFactory(ABC):
    """Contrato comum das factories de perda."""

    @abstractmethod
    def build(self, cfg):
        """Instancia o objeto de tarefa a partir de um TrainConfig ou EvalConfig."""


class LossFactoryRegistry:
    _factories: dict[str, LossFactory] = {}

    @staticmethod
    def _carregar_registros() -> None:
        """Importa o pacote, o que executa os ``_factories[...] = ...`` dos módulos.

        Sem isto o registry fica vazio quando alguém importa só a factory — que é o que o
        ``TrainEngine`` faz. O import fica aqui dentro, e não no topo, porque o módulo da
        factory é importado *pelos* modelos: no topo seria circular.
        """
        import importlib
        importlib.import_module("src.losses")

    @classmethod
    def get(cls, cfg) -> LossFactory:
        nome = resolve_task_name(cfg)
        cls._carregar_registros()
        if not cls._factories:
            raise NotImplementedError(
                "nenhuma perda registrada ainda — as perdas da Parte 2 entram em "
                "src/losses/. As partes 0 e 1 rodam sem treino."
            )
        if nome not in cls._factories:
            raise ValueError(
                f"loss desconhecida: {nome!r}. Disponíveis: {sorted(cls._factories)}"
            )
        return cls._factories[nome]

    @classmethod
    def build(cls, cfg):
        """Atalho: escolhe a factory e já instancia."""
        return cls.get(cfg).build(cfg)
