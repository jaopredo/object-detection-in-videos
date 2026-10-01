"""Parte 1 — o baseline por quadro, e a quantificação do fracasso dele.

O que esta parte tem que produzir, pelo enunciado:

1. rastreamento com a fonte de detecções **congelada** (detecção pública ou detector
   pré-treinado, sem treinar nada);
2. métricas de trajetória: IDF1, ID switches, fragmentações e erro de contagem de
   identidades únicas;
3. a regra de associação e a gestão de nascimento/morte documentadas;
4. **o gráfico obrigatório do descolamento** — dois painéis sobre as mesmas sequências: em
   cima mAP e IDF1, embaixo identidades previstas ÷ verdadeiras e ID switches por identidade
   verdadeira, com as sequências ordenadas por um eixo de dificuldade.

O painel de cima e o de baixo medem coisas que **não conversam**: o de cima é do detector,
que está congelado; o de baixo é da associação, que é nossa. A distância entre eles é o
argumento do trabalho inteiro — a detecção está resolvida e a identidade não.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import AppConfig
from src.data.mot17 import SEQUENCIAS
from src.data.pipeline import DataPipeline
from src.evaluation.dificuldade import dificuldade
from src.metrics.detection import average_precision
from src.metrics.identity import evaluate
from src.tracking.iou_tracker import IoUTracker

#: variantes rodadas lado a lado para mostrar que "regras diferentes dão números
#: diferentes". A primeira é a do config; as outras mexem num botão por vez, de modo que a
#: diferença entre duas linhas tenha uma causa só.
VARIANTES = {
    "guloso": {"associacao": "greedy"},
    "limiar 0,5": {"iou_threshold": 0.5},
    "max_age 5": {"max_age": 5},
    "velocidade": {"velocidade": True},
}


@dataclass
class ResultadoSequencia:
    nome: str
    n_frames: int
    camera: str
    ap: float
    metricas: dict
    dificuldade: dict

    @property
    def movimento(self) -> float:
        """O eixo de dificuldade: deslocamento mediano em unidades de altura da caixa."""
        return self.dificuldade["movimento"]

    def linha(self) -> dict:
        return {"name": self.nome, "n_frames": self.n_frames, "camera": self.camera,
                "ap": self.ap, **self.dificuldade, **self.metricas}


def rastrear_e_medir(gt, dets, **kwargs) -> tuple[dict, float]:
    """Roda o rastreador ingênuo numa sequência e devolve (métricas, AP das detecções)."""
    tracker = IoUTracker(**kwargs)
    pred = tracker.run(dets, gt.name, gt.fps, gt.width, gt.height)
    # `min_visibility=0.0` na AP: um objeto totalmente ocluído não põe pixel na tela, e
    # contá-lo como falso negativo mediria a oclusão da cena em vez do detector
    ap = average_precision(dets, gt, min_visibility=0.0)["ap"]
    return evaluate(gt, pred).to_dict(), ap


class BaselineRunner:
    """Executa a Parte 1 num split e grava métricas, tabela de variantes e a figura."""

    def __init__(self, app_config: AppConfig, split: str | None = None):
        self.app_config = app_config
        self.cfg = app_config.get_eval_config()
        self.split = split or self.cfg.split
        self.tracking = dict(self.cfg.tracking)
        self.tracking.pop("tracker", None)

    def run(self) -> dict:
        if self.split == "test":
            print("\n*** avaliando no CONJUNTO DE TESTE ***")
            print("    Só deve acontecer uma vez, no fim.\n")

        dataset = DataPipeline(self.app_config, config=self.cfg).build_split(self.split)

        resultados = []
        for i in range(len(dataset)):
            gt = dataset[i]
            dets = dataset.detections(i)
            metricas, ap = rastrear_e_medir(gt, dets, **self.tracking)
            info = SEQUENCIAS.get(gt.name)
            resultados.append(ResultadoSequencia(
                nome=gt.name, n_frames=len(gt),
                camera=info.camera if info else "?",
                ap=ap, metricas=metricas, dificuldade=dificuldade(gt),
            ))
            self._imprimir_sequencia(resultados[-1])

        resultados.sort(key=lambda r: r.movimento)
        variantes = self._rodar_variantes(dataset)

        out = Path(self.cfg.output_dir) / self.split
        out.mkdir(parents=True, exist_ok=True)
        (out / "per_sequence.json").write_text(
            json.dumps([r.linha() for r in resultados], indent=2), encoding="utf-8")

        resumo = self._resumir(resultados)
        resumo["variantes"] = variantes
        resumo["tracking"] = self.tracking
        resumo["detector"] = self.cfg.data.get("detector", "SDP")
        (out / "summary.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")

        figura = Path(self.cfg.figures_dir) / f"p1_descolamento_{self.split}.png"
        self._figura_descolamento(resultados, figura)

        self._imprimir_resumo(resumo, variantes, out, figura)
        return resumo

    # ------------------------------------------------------------------------ variantes

    def _rodar_variantes(self, dataset) -> dict:
        """Roda os mesmos vídeos com um botão mexido por vez."""
        print("\n  variantes da regra de associação (mesmos vídeos, um botão por vez)")
        saida = {}
        for nome, mudanca in {"config": {}, **VARIANTES}.items():
            kwargs = {**self.tracking, **mudanca}
            idf1, idsw, frag, erro = [], 0, 0, 0
            for i in range(len(dataset)):
                gt = dataset[i]
                m, _ = rastrear_e_medir(gt, dataset.detections(i), **kwargs)
                idf1.append(m["idf1"])
                idsw += m["id_switches"]
                frag += m["fragmentations"]
                erro += m["count_error"]
            saida[nome] = {"idf1": float(np.mean(idf1)), "id_switches": idsw,
                           "fragmentations": frag, "count_error": erro}
            print(f"    {nome:<12} IDF1 {saida[nome]['idf1']:.4f} | "
                  f"IDSW {idsw:>5} | FRAG {frag:>5} | erro cont. {erro:>4}")
        return saida

    # -------------------------------------------------------------------------- resumo

    def _resumir(self, resultados: list[ResultadoSequencia]) -> dict:
        def media(chave):
            return float(np.mean([r.metricas[chave] for r in resultados]))

        return {
            "split": self.split,
            "n_sequences": len(resultados),
            "n_frames": sum(r.n_frames for r in resultados),
            "ap_mean": float(np.mean([r.ap for r in resultados])),
            "idf1_mean": media("idf1"),
            "idp_mean": media("idp"),
            "idr_mean": media("idr"),
            "mota_mean": media("mota"),
            "id_switches": int(sum(r.metricas["id_switches"] for r in resultados)),
            "fragmentations": int(sum(r.metricas["fragmentations"] for r in resultados)),
            "n_gt_ids": int(sum(r.metricas["n_gt_ids"] for r in resultados)),
            "n_pred_ids": int(sum(r.metricas["n_pred_ids"] for r in resultados)),
            "count_error": int(sum(r.metricas["count_error"] for r in resultados)),
        }

    def _imprimir_sequencia(self, r: ResultadoSequencia) -> None:
        m = r.metricas
        print(f"  {r.nome:<11} {r.camera:<7} mov {r.movimento:>6.4f} | "
              f"AP {r.ap:.4f} | IDF1 {m['idf1']:.4f} | "
              f"IDSW {m['id_switches']:>4} | FRAG {m['fragmentations']:>4} | "
              f"ids {m['n_pred_ids']:>4}/{m['n_gt_ids']:<4}")

    def _imprimir_resumo(self, resumo, variantes, out, figura) -> None:
        print(f"\n  {resumo['n_sequences']} sequências | {resumo['n_frames']} quadros | "
              f"detector {resumo['detector']}\n")
        print("  DETECÇÃO (congelada — não muda mais no resto do PA)")
        print(f"    AP@0,5 média         {resumo['ap_mean']:.4f}")
        print("\n  IDENTIDADE (o que é nosso)")
        print(f"    IDF1 média           {resumo['idf1_mean']:.4f}")
        print(f"    ID switches          {resumo['id_switches']}")
        print(f"    fragmentações        {resumo['fragmentations']}")
        print(f"    identidades          {resumo['n_pred_ids']} previstas "
              f"para {resumo['n_gt_ids']} verdadeiras "
              f"({resumo['n_pred_ids'] / max(resumo['n_gt_ids'], 1):.1f}x)")
        print(f"\n  por sequência: {out / 'per_sequence.json'}")
        print(f"  resumo:        {out / 'summary.json'}")
        print(f"  figura:        {figura}")

    # -------------------------------------------------------------------------- figura

    def _figura_descolamento(self, resultados, path: Path) -> None:
        """O gráfico obrigatório: detecção em cima, identidade embaixo.

        As sequências vão no eixo x **ordenadas por densidade** — o eixo de dificuldade
        escolhido —, com a natureza da câmera anotada em cada rótulo. Ordenar importa: com
        as sequências em ordem alfabética as duas curvas viram ruído, e a pergunta "o que
        piora quando fica mais difícil?" não tem como ser lida do gráfico.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        rotulos = [f"{r.nome.replace('MOT17-', '')}\n{r.camera}\n{r.movimento:.4f}"
                   for r in resultados]
        x = np.arange(len(resultados))

        fig, (cima, baixo) = plt.subplots(
            2, 1, figsize=(max(7, 1.6 * len(resultados)), 7.2), sharex=True,
            gridspec_kw={"hspace": 0.12},
        )

        cima.plot(x, [r.ap for r in resultados], "o-", color="#4C72B0", lw=2,
                  label="AP@0,5 da detecção (congelada)")
        cima.plot(x, [r.metricas["idf1"] for r in resultados], "s-", color="#C44E52", lw=2,
                  label="IDF1 do rastreamento")
        for i, r in enumerate(resultados):
            cima.annotate("", xy=(i, r.ap), xytext=(i, r.metricas["idf1"]),
                          arrowprops=dict(arrowstyle="<->", color="gray", lw=0.9, alpha=0.6))
            meio = (r.ap + r.metricas["idf1"]) / 2
            cima.annotate(f"−{r.ap - r.metricas['idf1']:.2f}", xy=(i, meio),
                          xytext=(6, 0), textcoords="offset points",
                          fontsize=8, color="gray", va="center")
        cima.set_ylim(0, 1.05)
        cima.set_ylabel("qualidade")
        cima.legend(fontsize=9, loc="lower left")
        cima.grid(alpha=0.25)
        cima.set_title(
            "O descolamento: a detecção resolve, a identidade não\n"
            "sequências ordenadas por movimento (deslocamento por quadro ÷ altura da caixa)",
            fontsize=11,
        )

        razao = [r.metricas["n_pred_ids"] / max(r.metricas["n_gt_ids"], 1)
                 for r in resultados]
        por_id = [r.metricas["id_switches"] / max(r.metricas["n_gt_ids"], 1)
                  for r in resultados]

        baixo.bar(x - 0.2, razao, width=0.4, color="#DD8452",
                  label="identidades previstas ÷ verdadeiras")
        baixo.bar(x + 0.2, por_id, width=0.4, color="#8172B3",
                  label="ID switches por identidade verdadeira")
        baixo.axhline(1.0, color="gray", ls="--", lw=1,
                      label="1 = contagem de identidades certa")
        for i, (r_, p_) in enumerate(zip(razao, por_id)):
            baixo.annotate(f"{r_:.1f}x", xy=(i - 0.2, r_), xytext=(0, 3),
                           textcoords="offset points", ha="center", fontsize=8)
            baixo.annotate(f"{p_:.1f}", xy=(i + 0.2, p_), xytext=(0, 3),
                           textcoords="offset points", ha="center", fontsize=8)
        baixo.set_ylabel("razão / contagem")
        baixo.set_xticks(x)
        baixo.set_xticklabels(rotulos, fontsize=9)
        baixo.legend(fontsize=9)
        baixo.grid(alpha=0.25, axis="y")

        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
