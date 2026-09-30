"""Configuração de avaliação como dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    seed: int
    data: dict
    output_dir: Path
    # Mesmos opcionais do TrainConfig — ver src/training/config.py.
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)
    figures_dir: Path = Path("outputs/figures")
    #: em qual conjunto avaliar. O teste só deve ser tocado uma vez, no fim: a seleção de
    #: época e de hiperparâmetro acontece toda na validação (foi o que faltou no PA1).
    split: str = "val"

    @classmethod
    def from_dict(cls, cfg: dict) -> "EvalConfig":
        return cls(
            seed=cfg["seed"],
            data=cfg["data"],
            output_dir=Path(cfg["output_dir"]),
            model=cfg.get("model", {}),
            train=cfg.get("train", {}),
            tracking=cfg.get("tracking", {}),
            figures_dir=Path(cfg.get("figures_dir", "outputs/figures")),
            split=cfg.get("split", "val"),
        )
