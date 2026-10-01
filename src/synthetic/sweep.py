"""Parte 0, artefato 4 — o piso fácil e a varredura dos botões.

O enunciado pede duas coisas nesta parte, e são duas coisas diferentes:

**O baseline no piso fácil.** Poucas elipses, lentas, sem oclusão. *"O IDF1 tem que ficar
muito perto de 1."* É o teste de sanidade do conjunto métrica + rastreador: se falhar aqui,
qualquer número do MOT17 está errado e ninguém saberia dizer de onde.

**Depois, girem os botões.** Mais objetos, mais rápidos, oclusão mais longa. *"Mostrem onde
o baseline começa a quebrar. Esse gráfico é o ensaio da Parte 1."*

Um eixo por vez, partindo sempre da mesma configuração fácil — mexer em dois botões juntos
dá uma curva de onde não se lê qual dos dois causou o quê.

Uma decisão que mudou o resultado: o detector-oráculo recebe ``min_visibility=0.0``, ou
seja, **não vê o que está totalmente escondido**. A caixa do gabarito é amodal e existe
mesmo com o objeto invisível; servi-la como detecção cria um detector que enxerga através
das coisas. Medimos com ele: IDF1 = 1,0000 para oclusões de 0, 5, 10 e 20 quadros — a
oclusão não chegava ao rastreador e o botão não media nada.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import AppConfig
from src.data.detections import detections_from_sequence
from src.data.detector_sim import simulate_detections
from src.data.synthetic_video import SyntheticVideos
from src.metrics.identity import evaluate
from src.tracking import IoUTracker

#: a configuração do piso fácil: três elipses, devagar, sem poste.
PISO_FACIL = dict(
    n_sequences=4, seed=0, min_obj=3, max_obj=3, speed=1.0,
    occlusion_duration=0, n_occluders=0, min_frames=40, max_frames=40,
)

#: um eixo por vez. A chave diz qual parâmetro do gerador o valor substitui.
EIXOS = {
    "nº de objetos": ("objetos", [3, 6, 10, 15, 20]),
    "velocidade (px/quadro)": ("velocidade", [1.0, 2.0, 4.0, 6.0, 8.0]),
    "duração da oclusão (quadros)": ("oclusao", [0, 5, 10, 20, 30]),
    "detector: % descartado": ("drop", [0.0, 0.1, 0.2, 0.4, 0.6]),
}


def configuracao(eixo: str, valor) -> dict:
    cfg = dict(PISO_FACIL)
    if eixo == "objetos":
        cfg.update(min_obj=int(valor), max_obj=int(valor))
    elif eixo == "velocidade":
        cfg.update(speed=float(valor))
    elif eixo == "oclusao":
        cfg.update(occlusion_duration=int(valor), n_occluders=1 if valor else 0)
    return cfg


def medir(cfg_gerador: dict, drop_p: float = 0.0, **botoes) -> dict:
    """Roda o baseline num conjunto sintético e devolve as médias."""
    seqs = SyntheticVideos(**cfg_gerador)
    tracker = IoUTracker(**botoes)
    idf1, switches, frag, erro = [], 0, 0, 0

    for i in range(len(seqs)):
        s = seqs[i]
        if drop_p > 0:
            dets = simulate_detections(s, drop_p=drop_p, rng=i, min_visibility=0.0)
        else:
            dets = detections_from_sequence(s, min_visibility=0.0)
        m = evaluate(s, tracker.run(dets, s.name, s.fps, s.width, s.height))
        idf1.append(m.idf1)
        switches += m.id_switches
        frag += m.fragmentations
        erro += m.count_error

    return {"idf1": float(np.mean(idf1)), "id_switches": switches,
            "fragmentations": frag, "count_error": erro}


class SweepRunner:
    """Executa o piso fácil e os quatro eixos, e grava o gráfico do ensaio da Parte 1."""

    def __init__(self, app_config: AppConfig, saida: str | Path | None = None):
        self.cfg = app_config.get_train_config()
        self.saida = Path(saida) if saida else self.cfg.output_dir
        self.botoes = {k: v for k, v in (self.cfg.tracking or {}).items()
                       if k not in ("tracker", "velocidade", "emitir_sem_observacao")}
        self.botoes.setdefault("iou_threshold", 0.3)
        self.botoes.setdefault("max_age", 30)
        self.botoes.setdefault("min_hits", 1)

    def run(self) -> dict:
        print("PISO FÁCIL — 3 elipses, 1 px/quadro, sem oclusão, detecção perfeita")
        piso = medir(PISO_FACIL, **self.botoes)
        print(f"  IDF1 {piso['idf1']:.4f} | ID switches {piso['id_switches']} | "
              f"fragmentações {piso['fragmentations']} | erro de contagem {piso['count_error']}")
        veredito = "OK" if piso["idf1"] > 0.95 else "FALHOU — o bug é nosso, não do MOT17"
        print(f"  {veredito}\n")

        resultados = {}
        for rotulo, (eixo, valores) in EIXOS.items():
            print(f"{rotulo}")
            pontos = []
            for valor in valores:
                drop = float(valor) if eixo == "drop" else 0.0
                r = medir(configuracao(eixo, valor), drop_p=drop, **self.botoes)
                pontos.append({"valor": float(valor), **r})
                print(f"  {valor:>6} → IDF1 {r['idf1']:.4f} | IDSW {r['id_switches']:>4} "
                      f"| FRAG {r['fragmentations']:>4} | erro cont. {r['count_error']:>4}")
            resultados[rotulo] = pontos
            print()

        out = self.saida / "sweep"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(
            json.dumps({"piso_facil": piso, "eixos": resultados}, indent=2,
                       ensure_ascii=False), encoding="utf-8")

        figura = Path(self.cfg.figures_dir) / "p0_sweep.png"
        self._figura(piso, resultados, figura)
        print(f"detalhe: {out / 'sweep.json'}")
        print(f"figura:  {figura}")
        return {"piso_facil": piso, "eixos": resultados}

    def _figura(self, piso, resultados, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, eixos = plt.subplots(1, len(resultados), figsize=(3.6 * len(resultados), 3.8),
                                  sharey=True)
        eixos = np.atleast_1d(eixos)

        for ax, (rotulo, pontos) in zip(eixos, resultados.items()):
            x = [p["valor"] for p in pontos]
            ax.plot(x, [p["idf1"] for p in pontos], "o-", color="#C44E52", lw=2)
            ax.axhline(piso["idf1"], color="#55A868", ls="--", lw=1.2,
                       label=f"piso fácil = {piso['idf1']:.3f}")
            for p in pontos:
                if p["id_switches"]:
                    ax.annotate(f"{p['id_switches']} sw", xy=(p["valor"], p["idf1"]),
                                xytext=(0, -13), textcoords="offset points",
                                ha="center", fontsize=7, color="gray")
            ax.set_xlabel(rotulo, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
        eixos[0].set_ylabel("IDF1")
        eixos[0].legend(fontsize=8, loc="lower left")

        fig.suptitle(
            "Parte 0 — onde o baseline por IoU começa a quebrar (sintético, detecção perfeita)\n"
            "um botão por vez, partindo sempre do piso fácil",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
