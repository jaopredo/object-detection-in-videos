"""Configura o ambiente para que imports relativos ao projeto funcionem.

Coloca a pasta `deeplearning-net/` no sys.path de qualquer arquivo que o importe,
para que `from src.xxx import yyy` funcione independente de onde o script é rodado.
"""

import sys
from pathlib import Path


def setup_environment() -> None:
    """Adiciona a raiz do projeto (deeplearning-net/) ao sys.path."""
    # helpers/setup_environment.py -> helpers/ -> deeplearning-net/
    project_root = Path(__file__).resolve().parent.parent.parent
    str_root = str(project_root)
    if str_root not in sys.path:
        sys.path.insert(0, str_root)
