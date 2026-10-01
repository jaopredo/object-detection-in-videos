"""Inferência ponta a ponta: caminho de uma sequência → vídeo com identidades coloridas.

É o que o entregável ``notebooks/inferencia.ipynb`` usa. Mora aqui, e não no notebook, por
dois motivos: notebook não se testa, e o mesmo código serve à galeria da Parte 4.

Funciona em **qualquer pasta no formato MOTChallenge** — as sequências sintéticas da Parte 0
e as do MOT17 caem no mesmo caminho, porque o gerador da Parte 0 grava nesse formato de
propósito. Não precisa de gabarito: com ``det.txt`` e as imagens, já sai vídeo.
"""

from pathlib import Path

import numpy as np
import torch

from src.data.detections import read_mot_det
from src.data.mot17 import ler_seqinfo
from src.data.mot_format import read_mot_gt
from src.data.sequence import Sequence
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.tracking.rnn_tracker import RNNTracker

#: 20 cores bem separadas. A identidade escolhe a cor por ``id * 7 % 20``: multiplicar por um
#: número primo antes do módulo faz identidades consecutivas — que costumam estar lado a lado
#: na cena, porque nasceram no mesmo quadro — caírem em cores distantes.
N_CORES = 20


def cor_da_identidade(track_id: int) -> tuple[int, int, int]:
    """Cor RGB estável de uma identidade, em 0–255."""
    import matplotlib.pyplot as plt

    r, g, b, _ = plt.cm.tab20((int(track_id) * 7) % N_CORES)
    return int(r * 255), int(g * 255), int(b * 255)


def carregar_sequencia(pasta: str | Path) -> dict:
    """Lê uma pasta no formato MOTChallenge.

    Returns:
        ``{"nome", "info", "deteccoes", "gabarito"}``. O gabarito é ``None`` quando a pasta
        não tem ``gt/gt.txt`` — o que é o caso normal de um vídeo novo, e o rastreamento
        funciona sem ele.
    """
    pasta = Path(pasta)
    info = ler_seqinfo(pasta)

    det = pasta / "det" / "det.txt"
    if not det.exists():
        raise FileNotFoundError(
            f"{det} não existe. A pasta precisa do formato MOTChallenge: "
            f"det/det.txt, seqinfo.ini e img1/."
        )
    deteccoes = read_mot_det(det, n_frames=info["n_frames"])

    gt_path = pasta / "gt" / "gt.txt"
    gabarito = None
    if gt_path.exists():
        gabarito = read_mot_gt(
            gt_path, name=pasta.name, fps=info["fps"], width=info["width"],
            height=info["height"], n_frames=info["n_frames"],
            only_active_pedestrians=True,
        )

    return {"nome": pasta.name, "pasta": pasta, "info": info,
            "deteccoes": deteccoes, "gabarito": gabarito}


def carregar_modelo(config_path: str | Path, checkpoint: str | Path):
    """Reconstrói o modelo a partir do YAML e carrega os pesos. **Não treina nada.**"""
    from src.ablation.runner import StandaloneConfig
    import yaml

    cfg = StandaloneConfig(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    model = ModelFactoryRegistry.build(cfg.get_train_config())
    load_checkpoint(checkpoint, model, "cpu")
    return model.eval(), cfg


@torch.no_grad()
def rastrear(pasta: str | Path, model, botoes: dict) -> dict:
    """Roda o rastreador recorrente numa sequência e devolve as trajetórias.

    Returns:
        ``{"predicao": Sequence, "n_identidades": int, ...}``. ``n_identidades`` é a
        contagem de objetos únicos no vídeo — o entregável pedido pelo enunciado.
    """
    dados = carregar_sequencia(pasta)
    info = dados["info"]
    predicao = RNNTracker(model, **botoes).run(
        dados["deteccoes"], dados["nome"], info["fps"], info["width"], info["height"]
    )
    return {
        **dados,
        "predicao": predicao,
        "n_identidades": int(len(predicao.track_ids)),
        "n_deteccoes": int(sum(len(d) for d in dados["deteccoes"])),
    }


def desenhar(imagem: np.ndarray, frame, espessura: int = 3) -> np.ndarray:
    """Desenha as caixas de um quadro, cada identidade na sua cor, sem matplotlib.

    Desenho direto no array para poder gravar milhares de quadros: abrir uma figura por
    quadro seria ordens de grandeza mais lento e produziria bordas brancas no vídeo.
    """
    saida = imagem.copy()
    altura, largura = saida.shape[:2]

    for track_id, caixa in zip(frame.ids, frame.boxes):
        c = cor_da_identidade(int(track_id))
        x1, y1, x2, y2 = (int(round(v)) for v in caixa)
        x1, x2 = max(0, min(x1, largura - 1)), max(0, min(x2, largura - 1))
        y1, y2 = max(0, min(y1, altura - 1)), max(0, min(y2, altura - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        for k in range(espessura):
            if y1 + k < altura:
                saida[y1 + k, x1:x2] = c
            if y2 - k >= 0:
                saida[y2 - k, x1:x2] = c
            if x1 + k < largura:
                saida[y1:y2, x1 + k] = c
            if x2 - k >= 0:
                saida[y1:y2, x2 - k] = c
        # etiqueta: uma barra cheia acima da caixa, na cor da identidade. Não é o número —
        # é a cor que carrega a identidade, e ela é estável ao longo do vídeo, que é o que
        # o enunciado pede que se veja.
        topo = max(0, y1 - 10)
        saida[topo:y1, x1:min(x1 + 34, largura)] = c
    return saida


def gravar_video(resultado: dict, destino: str | Path, fps: float | None = None) -> Path:
    """Grava o mp4 com as identidades coloridas de forma consistente."""
    import imageio.v2 as imageio

    pasta, info = resultado["pasta"], resultado["info"]
    predicao: Sequence = resultado["predicao"]
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    quadros = []
    for t in range(len(predicao)):
        caminho = pasta / info["im_dir"] / f"{t + 1:06d}{info['im_ext']}"
        if not caminho.exists():
            raise FileNotFoundError(
                f"{caminho} não existe — o vídeo precisa das imagens. "
                f"Rode `make dados-imagens` (5,86 GB) ou aponte para uma sequência sintética."
            )
        quadros.append(desenhar(imageio.imread(caminho), predicao[t]))

    imageio.mimwrite(destino, quadros, fps=fps or info["fps"], macro_block_size=1)
    return destino
