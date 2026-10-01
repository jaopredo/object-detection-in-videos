"""Constrói os conjuntos de treino, validação e teste a partir da configuração.

Mesmo padrão do PA1: o registry escolhe a fonte pela chave ``data.kind``, e cada factory
declara em ``CHAVES`` o que ela realmente lê do bloco ``data``. Um ``occlusion_duracion``
com typo, ou um parâmetro do MOT17 deixado num config sintético, falha na hora em vez de
ser ignorado em silêncio.

A diferença para o PA1 são **três** conjuntos em vez de dois. No PA1 não houve validação:
os hiperparâmetros eram escolhidos à mão e o mesmo conjunto servia de validação e de
resultado final. Aqui os três papéis ficam separados — época e hiperparâmetro se decidem na
validação, e o teste só é tocado no fim.
"""

from abc import ABC, abstractmethod

from src.core.config import AppConfig
from src.training.config import TrainConfig

#: deslocamentos de seed por conjunto. Números arbitrários, mas fixos: garantem que
#: treino, validação e teste sejam vídeos diferentes, e que continuem sendo os mesmos
#: vídeos em toda execução com a mesma seed.
_OFFSET_SEED = {"train": 0, "val": 777, "test": 1337}


class DatasetFactory(ABC):
    #: chaves que esta factory realmente lê de ``data``. Serve para o config recusar
    #: chaves que ninguém consome.
    CHAVES: set[str] = set()

    @abstractmethod
    def build(self, cfg: TrainConfig, split: str):
        """Constrói o conjunto de um dos splits: ``train``, ``val`` ou ``test``."""

    def build_train(self, cfg: TrainConfig):
        return self.build(cfg, "train")

    def build_val(self, cfg: TrainConfig):
        return self.build(cfg, "val")

    def build_test(self, cfg: TrainConfig):
        return self.build(cfg, "test")

    def validate(self, cfg: TrainConfig) -> None:
        """Recusa chaves de ``data`` que esta factory não consome.

        Falhar aqui é barato; descobrir depois de rodar meia hora que o parâmetro estava
        sendo ignorado, não.
        """
        desconhecidas = set(cfg.data) - self.CHAVES - {"kind"}
        if desconhecidas:
            raise ValueError(
                f"chaves de 'data' que {type(self).__name__} não usa: "
                f"{sorted(desconhecidas)}. Aceitas: {sorted(self.CHAVES | {'kind'})}"
            )


class SyntheticVideoFactory(DatasetFactory):
    """Vídeos sintéticos da Parte 0.

    Os três conjuntos são gerados pelo mesmo código com seeds diferentes, então são vídeos
    genuinamente distintos e não recortes de um mesmo pool — o que já respeita a regra do
    enunciado de nunca separar por quadro, e sim por sequência.
    """

    CHAVES = {
        "size", "n_train", "n_val", "n_test", "min_frames", "max_frames",
        "min_obj", "max_obj", "speed", "occlusion_duration", "n_occluders",
        "noise", "fps",
    }

    def build(self, cfg: TrainConfig, split: str):
        from src.data.synthetic_video import SyntheticVideos

        if split not in _OFFSET_SEED:
            raise ValueError(f"split desconhecido: {split!r}. Use train, val ou test.")

        d = cfg.data
        return SyntheticVideos(
            n_sequences=d[f"n_{split}"],
            size=d.get("size", 128),
            seed=cfg.seed + _OFFSET_SEED[split],
            min_frames=d.get("min_frames", 30),
            max_frames=d.get("max_frames", 60),
            min_obj=d.get("min_obj", 5),
            max_obj=d.get("max_obj", 15),
            speed=d.get("speed", 2.0),
            occlusion_duration=d.get("occlusion_duration", 10),
            n_occluders=d.get("n_occluders", 1),
            noise=d.get("noise", 0.05),
            fps=d.get("fps", 30.0),
            # nome do split por extenso: abreviar pela inicial faria 'train' e 'test'
            # colidirem em 'T', e as sequências de um sobrescreveriam as do outro no disco
            prefix=f"SYNTH-{split.upper()}",
        )


class MOT17Factory(DatasetFactory):
    """O MOT17 lido do disco, com o split **por sequência** fixado em ``src/data/mot17.py``.

    Ao contrário do sintético, aqui os três conjuntos não são gerados: são escolhas de quais
    das 7 sequências de ``train/`` entram em cada um. O ``seed`` não influencia nada — o
    split é fixo de propósito, para que dois experimentos rodados em dias diferentes estejam
    falando das mesmas sequências.

    O ``test/`` do benchmark não entra em lugar nenhum: ele não tem ``gt.txt``.
    """

    CHAVES = {"root", "detector", "min_score"}

    def build(self, cfg: TrainConfig, split: str):
        from src.data.mot17 import MOT17Split

        return MOT17Split(
            root=cfg.data.get("root", "src/data/datasets/MOT17"),
            split=split,
            detector=cfg.data.get("detector", "SDP"),
        )


class DatasetFactoryRegistry:
    _factories = {
        "synthetic_video": SyntheticVideoFactory(),
        "mot17": MOT17Factory(),
    }

    @classmethod
    def get(cls, app_config: AppConfig) -> DatasetFactory:
        cfg = app_config.get_train_config()
        kind = cfg.data["kind"]
        if kind not in cls._factories:
            raise ValueError(
                f"data.kind desconhecido: {kind!r}. Disponíveis: {sorted(cls._factories)}"
            )
        factory = cls._factories[kind]
        factory.validate(cfg)
        return factory
