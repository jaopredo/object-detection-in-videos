"""Configuração de treino como dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int
    data: dict
    output_dir: Path
    # Blocos que nem toda parte do assignment usa. A Parte 0 é só geração de dados: não
    # tem modelo, não tem treino e não tem associação. Deixá-los opcionais evita encher o
    # YAML de blocos vazios só para o parser aceitar.
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    # parâmetros da associação e da gestão de tracks (limiar de IoU, max_age, min_hits).
    # É o análogo do bloco `decode` do PA1: o que acontece depois da rede, sem treino.
    tracking: dict = field(default_factory=dict)
    # onde caem os gráficos. Runs normais usam o padrão; a ablação sobrescreve para a
    # pasta da seed, mantendo tudo sob outputs/ablation/<timestamp>.
    figures_dir: Path = Path("outputs/figures")

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainConfig":
        return cls(
            seed=cfg["seed"],
            data=cfg["data"],
            output_dir=Path(cfg["output_dir"]),
            model=cfg.get("model", {}),
            train=cfg.get("train", {}),
            tracking=cfg.get("tracking", {}),
            figures_dir=Path(cfg.get("figures_dir", "outputs/figures")),
        )
