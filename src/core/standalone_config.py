"""Config não-singleton, para várias configurações na mesma sessão de processo.

``AppConfig`` é singleton (``SingletonMeta``): a segunda chamada devolve a primeira instância
e ignora o caminho novo. Numa grade de runs (ablação, grid search) isso faria todas as
combinações rodarem com o config da primeira, e os números sairiam plausíveis e errados.
"""

from pathlib import Path

from src.evaluation.config import EvalConfig
from src.training.config import TrainConfig


class StandaloneConfig:
    def __init__(self, raw: dict):
        self._raw = raw
        self.config_path = Path(raw.get("_origem", "<grade>"))
        self.train_config = TrainConfig.from_dict(raw)
        self.eval_config = EvalConfig.from_dict(raw)

    def get_train_config(self):
        return self.train_config

    def get_eval_config(self):
        return self.eval_config
