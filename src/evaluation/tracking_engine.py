"""Avaliação do modelo temporal — o que ``--mode eval`` faz a partir da Parte 2.

Roda o rastreador recorrente e os baselines **nas mesmas sequências, com a mesma fonte de
detecção e a mesma gestão de tracks**, e tabula os dois lado a lado. É o que o enunciado
pede na Parte 2: *"mostrem a comparação com as mesmas métricas lado a lado com um baseline,
nas mesmas sequências"*.

Igualar tudo que não é o objeto do experimento é o que dá sentido à diferença. Os dois
rastreadores compartilham ``TrackManager`` inteiro; muda só de onde sai a caixa prevista.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.config import AppConfig
from src.data.mot17 import SEQUENCIAS
from src.data.pipeline import DataPipeline
from src.evaluation.dificuldade import dificuldade
from src.metrics.detection import average_precision
from src.metrics.identity import evaluate
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.tracking.rnn_tracker import RNNTracker
from src.tracking import IoUTracker


class TrackingEvalEngine:
    """Compara o rastreador recorrente com os baselines num split."""

    def __init__(self, app_config: AppConfig, checkpoint: Path | None = None,
                 split: str | None = None):
        self.app_config = app_config
        self.cfg = app_config.get_eval_config()
        self.split = split or self.cfg.split
        self.checkpoint = Path(checkpoint) if checkpoint else Path(self.cfg.output_dir) / "best.pth"
        self.botoes = {k: v for k, v in self.cfg.tracking.items() if k != "tracker"}

    def _carregar_modelo(self):
        cfg = self.app_config.get_train_config()
        model = ModelFactoryRegistry.build(cfg)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"{self.checkpoint} não existe. Treine antes: "
                f"`python -m main --mode train --config {self.app_config.config_path}`"
            )
        estado = load_checkpoint(self.checkpoint, model, "cpu")
        print(f"checkpoint {self.checkpoint} — época {estado.get('epoch', '?')}, "
              f"{estado.get('monitor', 'val_loss')} {estado.get('best_metric', float('nan')):.5f}")
        return model.eval()

    def rastreadores(self, model) -> dict:
        """Os três, com **os mesmos** botões de associação e gestão de tracks.

        A ordem importa para a leitura: posição constante é o baseline do enunciado,
        velocidade é o baseline honesto (um modelo de movimento que não aprendeu nada), e a
        recorrência tem que ganhar dos dois para a Parte 2 significar alguma coisa.
        """
        return {
            "IoU (posição constante)": IoUTracker(**self.botoes),
            "IoU + velocidade": IoUTracker(velocidade=True, **self.botoes),
            "RNN (modelo de movimento)": RNNTracker(model, **self.botoes),
        }

    @torch.no_grad()
    def run(self) -> dict:
        if self.split == "test":
            print("\n*** avaliando no CONJUNTO DE TESTE ***")
            print("    Só deve acontecer uma vez, no fim.\n")

        model = self._carregar_modelo()
        dataset = DataPipeline(self.app_config, config=self.cfg).build_split(self.split)

        sequencias = []
        for i in range(len(dataset)):
            gt = dataset[i]
            sequencias.append((gt, dataset.detections(i)))

        resultados = {}
        for rotulo, tracker in self.rastreadores(model).items():
            por_sequencia = []
            for gt, dets in sequencias:
                pred = tracker.run(dets, gt.name, gt.fps, gt.width, gt.height)
                m = evaluate(gt, pred).to_dict()
                m["name"] = gt.name
                m["ap"] = average_precision(dets, gt, min_visibility=0.0)["ap"]
                m.update(dificuldade(gt))
                m["camera"] = SEQUENCIAS[gt.name].camera if gt.name in SEQUENCIAS else "?"
                por_sequencia.append(m)
            resultados[rotulo] = {
                "por_sequencia": por_sequencia,
                "resumo": self._resumir(por_sequencia),
            }

        out = Path(self.cfg.output_dir) / self.split
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparacao.json").write_text(
            json.dumps({"tracking": self.botoes, "modelo": model.config(),
                        "resultados": resultados}, indent=2), encoding="utf-8")

        figura = Path(self.cfg.figures_dir) / f"p2_comparacao_{self.split}.png"
        self._figura(resultados, figura)
        self._imprimir(resultados, out, figura)
        return resultados

    @torch.no_grad()
    def run_silencioso(self) -> dict:
        """Só o rastreador recorrente, sem figura e sem imprimir — para a ablação.

        Roda um dos três em vez dos três: os dois baselines não dependem do checkpoint, e
        recalculá-los em cada uma das 36 combinações da grade custaria dois terços do tempo
        para produzir sempre o mesmo número.
        """
        import contextlib
        import io
        import json as _json

        with contextlib.redirect_stdout(io.StringIO()):
            model = self._carregar_modelo()
            dataset = DataPipeline(self.app_config, config=self.cfg).build_split(self.split)
            tracker = RNNTracker(model, **self.botoes)
            por_sequencia = []
            for i in range(len(dataset)):
                gt = dataset[i]
                pred = tracker.run(dataset.detections(i), gt.name, gt.fps, gt.width, gt.height)
                m = evaluate(gt, pred).to_dict()
                m["name"] = gt.name
                por_sequencia.append(m)

        resumo = self._resumir(por_sequencia)
        resumo.pop("ap", None)           # a detecção está congelada; não varia na grade
        resumo["n_params"] = model.n_parametros()
        historico = Path(self.cfg.output_dir) / "history.json"
        if historico.exists():
            epocas = _json.loads(historico.read_text())
            resumo["val_loss"] = min(e["val_loss"] for e in epocas)
        return resumo

    @staticmethod
    def _resumir(por_sequencia: list[dict]) -> dict:
        media = lambda k: float(np.mean([s[k] for s in por_sequencia if k in s]))
        soma = lambda k: int(sum(s[k] for s in por_sequencia))
        return {
            "idf1": media("idf1"), "idp": media("idp"), "idr": media("idr"),
            "mota": media("mota"),
            "ap": media("ap") if "ap" in por_sequencia[0] else 0.0,
            "id_switches": soma("id_switches"), "fragmentations": soma("fragmentations"),
            "n_pred_ids": soma("n_pred_ids"), "n_gt_ids": soma("n_gt_ids"),
            "count_error": soma("count_error"),
        }

    def _imprimir(self, resultados, out, figura) -> None:
        print(f"\n  split {self.split} | {len(next(iter(resultados.values()))['por_sequencia'])} "
              f"sequências | detecção congelada (AP "
              f"{next(iter(resultados.values()))['resumo']['ap']:.4f})\n")
        print(f"  {'rastreador':<28} {'IDF1':>7} {'IDP':>6} {'IDR':>6} "
              f"{'IDSW':>6} {'FRAG':>6} {'ids':>10} {'MOTA':>7}")
        base = None
        for rotulo, r in resultados.items():
            s = r["resumo"]
            base = base if base is not None else s["idf1"]
            print(f"  {rotulo:<28} {s['idf1']:>7.4f} {s['idp']:>6.3f} {s['idr']:>6.3f} "
                  f"{s['id_switches']:>6} {s['fragmentations']:>6} "
                  f"{s['n_pred_ids']:>4}/{s['n_gt_ids']:<5} {s['mota']:>7.4f}")
        melhor = max(resultados.items(), key=lambda kv: kv[1]["resumo"]["idf1"])
        print(f"\n  melhor: {melhor[0]} — IDF1 {melhor[1]['resumo']['idf1']:.4f} "
              f"({100 * (melhor[1]['resumo']['idf1'] / base - 1):+.1f}% sobre o baseline)")
        print(f"\n  detalhe: {out / 'comparacao.json'}")
        print(f"  figura:  {figura}")

    def _figura(self, resultados, path: Path) -> None:
        """Quatro painéis: o que melhora, o que piora, e onde.

        IDF1 e ID switches são o que o enunciado pede. IDP e IDR entram porque **a
        diferença entre os dois é o diagnóstico**: um rastreador que sustenta a caixa no
        buraco ganha revocação e perde precisão, e o IDF1 sozinho esconde qual dos dois se
        mexeu.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        rotulos = list(resultados)
        nomes = [s["name"].replace("MOT17-", "") for s in resultados[rotulos[0]]["por_sequencia"]]
        cores = ["#4C72B0", "#DD8452", "#C44E52"]

        fig, eixos = plt.subplots(2, 2, figsize=(11.5, 7.5))
        painel = [
            ("idf1", "IDF1 (mais é melhor)", False),
            ("id_switches", "ID switches (menos é melhor)", True),
            ("idp", "IDP — precisão de identidade", False),
            ("idr", "IDR — revocação de identidade", False),
        ]

        largura = 0.8 / len(rotulos)
        x = np.arange(len(nomes))
        for ax, (chave, titulo, inteiro) in zip(eixos.ravel(), painel):
            for k, rotulo in enumerate(rotulos):
                valores = [s[chave] for s in resultados[rotulo]["por_sequencia"]]
                ax.bar(x + (k - (len(rotulos) - 1) / 2) * largura, valores,
                       width=largura, color=cores[k % len(cores)], label=rotulo)
            ax.set_xticks(x)
            ax.set_xticklabels(nomes, fontsize=9)
            ax.set_title(titulo, fontsize=10)
            ax.grid(alpha=0.25, axis="y")
            if not inteiro:
                ax.set_ylim(0, 1.0)

        eixos[0, 0].legend(fontsize=8, loc="upper left")
        fig.suptitle(
            "Parte 2 — mesma detecção, mesma gestão de tracks, muda só o modelo de movimento",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
