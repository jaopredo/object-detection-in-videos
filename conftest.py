"""Põe a raiz do projeto no ``sys.path`` para toda a suíte.

O pytest já acrescenta ``tests/`` sozinho (é o primeiro diretório sem ``__init__.py`` subindo
a partir do arquivo de teste), o que faz ``from helpers.x import y`` funcionar. O que ele
**não** acrescenta é a raiz, de onde saem ``src/`` e ``metrics.py`` — daí cada teste antigo
começar com quatro linhas de ``sys.path.insert``.

Um ``conftest.py`` na raiz é carregado pelo pytest antes de qualquer coleta, então essas
quatro linhas deixam de ser necessárias em todo arquivo novo.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
