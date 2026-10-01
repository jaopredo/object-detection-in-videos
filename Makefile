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

DADOS := src/data/datasets

# Anotações do MOT17 (~10 MB). É o que destrava as Partes 0 a 5 inteiras: a trilha A opera
# sobre caixas, e gt.txt + det.txt são texto.
dados:
	mkdir -p $(DADOS)
	curl -L -o $(DADOS)/MOT17Labels.zip https://motchallenge.net/data/MOT17Labels.zip
	cd $(DADOS) && unzip -q -o MOT17Labels.zip && mkdir -p MOT17 && mv train test MOT17/ && rm MOT17Labels.zip

# O pacote completo (5,86 GB). Só é preciso para as figuras da Parte 4 e para o vídeo do
# notebook. Extrai apenas MOT17-*-SDP/img1 (857 MB): as imagens são idênticas nas três
# variantes de detector, então extrair as 21 pastas triplicaria o disco por nada.
dados-imagens:
	mkdir -p $(DADOS)
	curl -L -C - -o $(DADOS)/MOT17.zip https://motchallenge.net/data/MOT17.zip
	cd $(DADOS) && unzip -q -o MOT17.zip 'MOT17/train/MOT17-*-SDP/img1/*'

# Parte 0 — vídeos sintéticos, gt.txt em formato MOT e a figura da oclusão.
parte0:
	python -m main --mode gen-synth --synthetic
	python -m main --mode sweep --synthetic

# Parte 1 — baseline por quadro no MOT17.
parte1:
	python -m main --mode baseline --config configs/mot17.yaml --split trainval

# Parte 2 — um comando que treina, um comando que avalia.
treinar:
	python -m main --mode train --config configs/mot17_gru.yaml

avaliar:
	python -m main --mode eval --config configs/mot17_gru.yaml

parte3:
	python -m main --mode ablation --config configs/mot17_gru.yaml

parte4:
	python -m main --mode fails --config configs/mot17_gru.yaml --split trainval

parte5:
	python -m main --mode stress --config configs/mot17_gru.yaml

test:
	pytest tests/ -q

.PHONY: setup setup-cpu setup-pip setup-cpu-pip dados dados-imagens \
        parte0 parte1 treinar avaliar parte3 parte4 parte5 test
