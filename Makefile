setup:
	uv sync
	uv pip install torch torchvision --torch-backend=auto

setup-cpu:
	uv sync
	uv pip install torch torchvision --torch-backend=cpu

# Regera requirements.txt a partir do pyproject.toml/uv.lock. Precisa de uv; é só para
# manutenção do arquivo versionado, não faz parte do fluxo de instalação sem uv.
requirements.txt:
	uv export --no-hashes --format requirements.txt -o requirements.txt

# Equivalente a `make setup`, mas sem uv: usa venv + pip padrão e o requirements.txt versionado.
setup-pip:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install torch torchvision

# Equivalente a `make setup-cpu`, mas sem uv.
setup-cpu-pip:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Parte 0 — gera os vídeos sintéticos, o gt.txt em formato MOT e a figura da oclusão.
parte0:
	python -m main --mode gen-synth --synthetic

test:
	pytest tests/ -v

.PHONY: setup setup-cpu setup-pip setup-cpu-pip parte0 test
