"""Motor de avaliação: roda sobre um dos splits e grava métricas e figuras.

Herdado do PA1 (``semantic-segmentation/src/evaluation/engine.py``): mesma forma — carrega
o conjunto, percorre, grava ``per_*.json`` e ``summary.json``, produz as figuras, imprime o
resumo. Duas diferenças:

**O split é escolhido, não implícito.** No PA1 havia só treino e validação, e a "avaliação"
sempre rodava na validação, que era também o número reportado. Aqui ``EvalConfig.split``
diz onde rodar, e tocar o teste é uma decisão explícita que fica registrada no log — a
seleção de época e de hiperparâmetro tem que acontecer toda na validação.

**Nesta fatia, o que se avalia é o dataset.** As métricas de rastreamento (IDF1, ID
switches, fragmentações) e o tracker que as alimenta chegam na próxima fatia. O que já dá
para medir, e que a Parte 4 vai precisar, é a caracterização das sequências: densidade,
número de identidades e **a distribuição de duração de oclusão** — o dado contra o qual o
horizonte de memória do modelo vai ser comparado.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import AppConfig
from src.data.pipeline import DataPipeline
from src.evaluation.occlusion import sequence_stats


class EvalEngine:
    def __init__(self, app_config: AppConfig, checkpoint: Path | None = None):
        self.app_config = app_config
        self.cfg = app_config.get_eval_config()
        self.checkpoint = checkpoint

    def run(self) -> dict:
        cfg = self.cfg
        split = cfg.split

        if split == "test":
            print("\n*** avaliando no CONJUNTO DE TESTE ***")
            print("    Só deve acontecer uma vez, no fim. Escolha de época e de")
            print("    hiperparâmetro se faz na validação.\n")

        dataset = DataPipeline(self.app_config, config=cfg).build_split(split)
        por_sequencia = [sequence_stats(dataset[i]) for i in range(len(dataset))]

        duracoes = [d for s in por_sequencia for d in s["occlusion_durations"]]
        resumo = {
            "split": split,
            "n_sequences": len(por_sequencia),
            "n_frames": int(sum(s["n_frames"] for s in por_sequencia)),
            "n_identities": int(sum(s["n_identities"] for s in por_sequencia)),
            "density_mean": float(np.mean([s["density_mean"] for s in por_sequencia])),
            "visibility_mean": float(np.mean([s["visibility_mean"] for s in por_sequencia])),
            "n_occlusions": len(duracoes),
            "occlusion_mean": float(np.mean(duracoes)) if duracoes else 0.0,
            "occlusion_p90": float(np.percentile(duracoes, 90)) if duracoes else 0.0,
            "occlusion_max": int(np.max(duracoes)) if duracoes else 0,
        }

        out_dir = cfg.output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_sequence.json").write_text(json.dumps(por_sequencia, indent=2))
        (out_dir / "summary.json").write_text(json.dumps(resumo, indent=2))

        figura = Path(cfg.figures_dir) / f"{split}_oclusao.png"
        self._plot_occlusion_hist(duracoes, figura, split)

        self._print_summary(resumo, out_dir, figura)
        return resumo

    def _plot_occlusion_hist(self, duracoes, path: Path, split: str) -> None:
        """Distribuição de duração de oclusão — a régua da Parte 4.

        É contra esta curva que o horizonte de memória do modelo vai ser comparado: se a
        janela de BPTT cobre 8 quadros e metade das oclusões dura mais que isso, o modelo
        nunca recebeu sinal de supervisão que atravessasse o buraco.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))

        if duracoes:
            maximo = max(duracoes)
            ax.hist(duracoes, bins=range(1, maximo + 2), align="left",
                    color="#4C72B0", alpha=0.85, edgecolor="white")
            ax.axvline(np.mean(duracoes), color="#C44E52", ls="--", lw=1.5,
                       label=f"média {np.mean(duracoes):.1f}")
            ax.axvline(np.percentile(duracoes, 90), color="#DD8452", ls=":", lw=1.5,
                       label=f"p90 {np.percentile(duracoes, 90):.0f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "nenhuma oclusão detectada", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

        ax.set_xlabel("duração da oclusão (quadros)")
        ax.set_ylabel("ocorrências")
        ax.set_title(f"Distribuição de duração de oclusão — {split}", fontsize=10)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)

    def _print_summary(self, resumo, out_dir, figura) -> None:
        print(f"\n{resumo['n_sequences']} sequências | {resumo['n_frames']} quadros "
              f"| {resumo['n_identities']} identidades\n")
        print("  DIFICULDADE DAS SEQUÊNCIAS")
        print(f"    densidade média      {resumo['density_mean']:.2f} objetos por quadro")
        print(f"    visibilidade média   {resumo['visibility_mean']:.3f}")
        print("\n  OCLUSÃO (a régua da Parte 4)")
        print(f"    trechos              {resumo['n_occlusions']}")
        print(f"    duração média        {resumo['occlusion_mean']:.1f} quadros")
        print(f"    duração p90          {resumo['occlusion_p90']:.0f} quadros")
        print(f"    duração máxima       {resumo['occlusion_max']} quadros")
        print(f"\n  por sequência: {out_dir/'per_sequence.json'}")
        print(f"  resumo:        {out_dir/'summary.json'}")
        print(f"  histograma:    {figura}")
