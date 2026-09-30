"""Singleton de configuração carregado via metaclass."""

import yaml
from pathlib import Path

from src.training.config import TrainConfig
from src.evaluation.config import EvalConfig


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AppConfig(metaclass=SingletonMeta):
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = Path(config_path)
        self._raw = yaml.safe_load(self.config_path.read_text())
        self.train_config = TrainConfig.from_dict(self._raw)
        self.eval_config = EvalConfig.from_dict(self._raw)

    def get_train_config(self) -> TrainConfig:
        return self.train_config

    def get_eval_config(self) -> EvalConfig:
        return self.eval_config
