"""A segunda fonte de detecção da Parte 1: um detector pré-treinado do torchvision.

O enunciado pede duas fontes e diz para usar as duas: as detecções públicas que acompanham o
MOT17, e *"um detector pré-treinado do torchvision em modo de inferência, classe person do
COCO"*. Fazer fine-tune é opcional e não vale ponto por si — aqui não é feito.

Duas restrições do enunciado moldam este módulo:

**O NMS é nosso.** ``torchvision.ops.nms`` é proibido, e o Faster R-CNN do torchvision aplica
NMS internamente antes de devolver as caixas. Por isso ele é construído com o NMS interno
praticamente desligado (``box_nms_thresh`` alto) e com o limiar de score baixo, e a supressão
de verdade é feita por ``src/tracking/nms.py``, sobre as propostas cruas.

**O resultado é cacheado em formato MOT.** Rodar o Faster R-CNN em CPU custa segundos por
quadro; a Parte 1 percorre milhares. Gravando um ``det_frcnn.txt`` ao lado do ``det.txt``
público, a saída vira indistinguível de uma fonte pública para o resto do projeto — e nenhuma
parte seguinte precisa rodar o detector de novo.

Sobre a resolução: o padrão do torchvision redimensiona o lado menor para 800 px. Em CPU,
1080 → 800 num Ryzen de 8 núcleos dá alguns segundos por quadro. ``min_size`` é parâmetro
para que a escolha entre tempo e qualidade seja explícita, e o valor usado fica gravado no
JSON ao lado das detecções.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.detections import Detections, write_mot_det
from src.data.mot17 import ler_seqinfo, pasta_da_sequencia
from src.tracking.nms import nms

#: índice da classe *person* no COCO, como o torchvision numera (0 é o fundo).
PESSOA = 1


def construir_detector(min_size: int = 800, score_minimo: float = 0.05):
    """Faster R-CNN pré-treinado, com o NMS interno praticamente desligado.

    O enunciado proíbe ``torchvision.ops.nms``, e o modelo o usa por dentro. Subindo
    ``box_nms_thresh`` para 0,95, quase nada é suprimido lá: o que sai são as propostas
    cruas, e a supressão de verdade fica com o nosso NMS. Não dá para zerar o uso interno
    sem reescrever a cabeça do modelo, e o enunciado permite ``torchvision.models.detection`` —
    o que ele pede é que a supressão do nosso pipeline seja nossa, e é.
    """
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                              fasterrcnn_resnet50_fpn)

    modelo = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        min_size=min_size, max_size=int(min_size * 16 / 9),
        box_score_thresh=score_minimo,
        box_nms_thresh=0.95,
        box_detections_per_img=300,
    )
    return modelo.eval()


@torch.no_grad()
def detectar_sequencia(
    root: str | Path,
    nome: str,
    detector=None,
    min_size: int = 800,
    iou_nms: float = 0.5,
    score_minimo: float = 0.3,
    limite_quadros: int | None = None,
    verbose: bool = True,
) -> list[Detections]:
    """Roda o detector em cada quadro de uma sequência e aplica o **nosso** NMS.

    Args:
        root, nome: onde está a sequência do MOT17.
        detector: modelo já construído; ``None`` constrói um.
        min_size: lado menor para o qual a imagem é redimensionada.
        iou_nms: limiar do nosso NMS.
        score_minimo: confiança mínima mantida depois do NMS.
        limite_quadros: roda só os primeiros N quadros. Existe porque o custo em CPU é real:
            com ele dá para medir o tempo por quadro antes de decidir rodar a sequência
            inteira.

    Returns:
        Uma lista com um ``Detections`` por quadro.
    """
    import imageio.v2 as imageio

    detector = detector or construir_detector(min_size)
    pasta = pasta_da_sequencia(root, nome)
    info = ler_seqinfo(pasta)
    total = min(info["n_frames"], limite_quadros or info["n_frames"])

    saida, t0 = [], time.perf_counter()
    for t in range(total):
        caminho = pasta / info["im_dir"] / f"{t + 1:06d}{info['im_ext']}"
        if not caminho.exists():
            raise FileNotFoundError(
                f"{caminho} não existe. O detector precisa das imagens: `make dados-imagens`."
            )

        imagem = torch.from_numpy(
            imageio.imread(caminho).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        bruto = detector([imagem])[0]

        pessoas = bruto["labels"] == PESSOA
        caixas = bruto["boxes"][pessoas].numpy()
        scores = bruto["scores"][pessoas].numpy()

        # NOSSO NMS, sobre as propostas que o modelo devolveu
        mantidas = nms(caixas, scores, iou_nms)
        caixas, scores = caixas[mantidas], scores[mantidas]
        acima = scores >= score_minimo

        saida.append(Detections(t, caixas[acima], scores[acima]))

        if verbose and (t + 1) % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"    {t + 1}/{total} quadros | {dt / (t + 1):.2f} s/quadro | "
                  f"faltam {(total - t - 1) * dt / (t + 1) / 60:.1f} min", flush=True)

    return saida


def cachear(
    root: str | Path, nome: str, destino: str | Path | None = None, **kwargs
) -> Path:
    """Roda o detector e grava o resultado em ``det_frcnn.txt``, em formato MOT.

    A partir daí, trocar de fonte de detecção é trocar um caminho de arquivo — nada no
    rastreador, nas métricas ou nas figuras precisa saber que uma delas veio de uma rede e a
    outra de um arquivo baixado.
    """
    pasta = pasta_da_sequencia(root, nome)
    destino = Path(destino) if destino else pasta / "det" / "det_frcnn.txt"

    print(f"  {nome}: rodando o Faster R-CNN (CPU)...", flush=True)
    t0 = time.perf_counter()
    dets = detectar_sequencia(root, nome, **kwargs)
    dt = time.perf_counter() - t0

    write_mot_det(dets, destino)
    meta = {
        "sequencia": nome, "n_quadros": len(dets),
        "n_deteccoes": int(sum(len(d) for d in dets)),
        "segundos": dt, "segundos_por_quadro": dt / max(len(dets), 1),
        "parametros": {k: v for k, v in kwargs.items() if k != "detector"},
    }
    destino.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  {nome}: {meta['n_deteccoes']} detecções em {dt / 60:.1f} min "
          f"({meta['segundos_por_quadro']:.2f} s/quadro) → {destino}")
    return destino
