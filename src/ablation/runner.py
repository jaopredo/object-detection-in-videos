"""Parte 3 — Eixo 1: a célula recorrente.

RNN simples × LSTM × GRU, **no mesmo orçamento de parâmetros**, variando o comprimento da
janela de BPTT truncado ``T ∈ {4, 8, 16, 32}``, com **3 seeds**. A pergunta específica do
enunciado: *onde a RNN simples quebra, e isso bate com a história de gradiente que some
contada na aula?*

Três decisões de método, todas herdadas do PA1:

**Orçamento igualado por busca, não por conta.** A RNN tem 1 porta, a GRU 3 e a LSTM 4; a
conta à mão é fácil de errar e ninguém confere. ``hidden_para_orcamento`` procura o maior
``hidden`` que cabe no teto, e o resultado é registrado no relatório (rnn 95, lstm 48,
gru 55 — entre 96,8% e 98,8% de 20k).

**3 seeds, e o efeito só conta se for maior que o espalhamento entre elas.** É a régua que
o PA1 usou para validar o Eixo 3, e é o que separa resultado de ruído.

**Grade de overrides, não 12 arquivos YAML.** Doze configs quase idênticos divergem — basta
alguém corrigir um e esquecer os outros. Aqui o YAML base é um só e a grade mexe em duas
chaves.

Os checkpoints ficam guardados porque **a Parte 4 os reaproveita**: o enunciado manda
comparar a curva de gradiente da RNN simples com a do modelo com portas na mesma janela, e
diz "os checkpoints já existem".
"""

import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

from src.core.grid_stats import mean_std_metrics
from src.core.standalone_config import StandaloneConfig

#: as três células do eixo e as quatro janelas de BPTT do enunciado
CELULAS = ("rnn", "lstm", "gru")
JANELAS = (4, 8, 16, 32)
SEEDS = (0, 1, 2)


class AblationRunner:
    """Executa a grade célula × janela × seed e relata média ± desvio."""

    def __init__(
        self,
        config_base: str,
        celulas=CELULAS,
        janelas=JANELAS,
        seeds=SEEDS,
        base_output_dir: str = "outputs/ablation",
    ):
        self.config_base = Path(config_base)
        self.celulas = tuple(celulas)
        self.janelas = tuple(janelas)
        self.seeds = tuple(seeds)
        self.base_output_dir = Path(base_output_dir)

    def run(self) -> dict:
        from src.evaluation.tracking_engine import TrackingEvalEngine
        from src.training.engine import TrainEngine

        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{carimbo}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        combinacoes = list(product(self.celulas, self.janelas, self.seeds))
        print(f"ablação Eixo 1 — {len(self.celulas)} células × {len(self.janelas)} janelas "
              f"× {len(self.seeds)} seeds = {len(combinacoes)} runs")
        print(f"saída: {self.run_dir}\n")

        resultados: dict = {}
        t0 = time.perf_counter()

        for n, (celula, janela, seed) in enumerate(combinacoes, start=1):
            chave = f"{celula}_T{janela}"
            destino = self.run_dir / chave / f"seed_{seed}"
            raw = self._config(celula, janela, seed, destino)
            app = StandaloneConfig(raw)

            print(f"[{n:>2}/{len(combinacoes)}] {celula:<4} T={janela:<2} seed={seed} ... ",
                  end="", flush=True)
            inicio = time.perf_counter()

            TrainEngine(app, resume=False).run_silencioso()
            metricas = TrackingEvalEngine(app, destino / "best.pth", split="val").run_silencioso()

            dt = time.perf_counter() - inicio
            resultados.setdefault(chave, {})[seed] = metricas
            print(f"IDF1 {metricas['idf1']:.4f} | IDSW {metricas['id_switches']:>4} "
                  f"| {dt:.0f}s")

        estatisticas = self._estatisticas(resultados)
        (self.run_dir / "report.json").write_text(
            json.dumps({"celulas": list(self.celulas), "janelas": list(self.janelas),
                        "seeds": list(self.seeds), "estatisticas": estatisticas},
                       indent=2), encoding="utf-8")
        self._relatorio(estatisticas)
        self._figura(estatisticas)

        total = time.perf_counter() - t0
        print(f"\ntempo total: {total / 60:.1f} min")
        print(f"relatório: {self.run_dir / 'report.md'}")
        print(f"figura:    {self.run_dir / 'eixo1.png'}")
        return estatisticas

    # ------------------------------------------------------------------------- config

    def _config(self, celula: str, janela: int, seed: int, destino: Path) -> dict:
        raw = yaml.safe_load(self.config_base.read_text(encoding="utf-8"))
        raw["seed"] = seed
        raw["model"]["cell"] = celula
        # `hidden` sai do config para a factory dimensionar a célula pelo orçamento — é o
        # que iguala os três braços do eixo
        raw["model"].pop("hidden", None)
        raw["train"]["window"] = janela
        raw["train"]["stride"] = max(1, janela // 2)
        # o buraco simulado não pode ser maior que a janela: com T=4 um buraco de 8 passos
        # mascararia a janela inteira e não sobraria alvo nenhum
        raw["train"]["oclusao_max"] = max(1, min(raw["train"].get("oclusao_max", 8), janela - 2))
        raw["output_dir"] = str(destino)
        raw["figures_dir"] = str(destino)
        raw["_origem"] = str(self.config_base)
        return raw

    # -------------------------------------------------------------------- estatísticas

    def _estatisticas(self, resultados: dict) -> dict:
        saida = {}
        for chave, por_seed in resultados.items():
            celula, janela = chave.split("_T")
            entrada = {"cell": celula, "window": int(janela),
                       "n_params": next(iter(por_seed.values())).get("n_params")}
            entrada.update(mean_std_metrics(
                por_seed,
                ("idf1", "idp", "idr", "id_switches", "fragmentations", "mota", "val_loss"),
            ))
            saida[chave] = entrada
        return saida

    def _relatorio(self, estatisticas: dict) -> None:
        linhas = [
            "# Parte 3 — Eixo 1: a célula recorrente",
            "",
            f"Gerado em {datetime.now():%Y-%m-%d %H:%M}. "
            f"{len(self.seeds)} seeds ({', '.join(map(str, self.seeds))}), "
            f"média ± desvio. Avaliado na **validação** (MOT17-10).",
            "",
            "| célula | T | parâmetros | IDF1 | ID switches | val_loss |",
            "|---|---|---|---|---|---|",
        ]
        for chave in sorted(estatisticas, key=lambda k: (k.split("_T")[0], int(k.split("_T")[1]))):
            e = estatisticas[chave]
            fmt = lambda m, c=4: (f"{e[m]['mean']:.{c}f} ± {e[m]['std']:.{c}f}"
                                  if m in e else "—")
            linhas.append(
                f"| {e['cell']} | {e['window']} | {e.get('n_params', '—')} | "
                f"{fmt('idf1')} | {fmt('id_switches', 1)} | {fmt('val_loss', 5)} |"
            )
        (self.run_dir / "report.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")

        print("\n" + "\n".join(linhas[4:]))

    def _figura(self, estatisticas: dict) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cores = {"rnn": "#C44E52", "lstm": "#4C72B0", "gru": "#55A868"}
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

        for celula in self.celulas:
            xs, ys, erros, sw, sw_err = [], [], [], [], []
            for janela in self.janelas:
                e = estatisticas.get(f"{celula}_T{janela}")
                if not e or "idf1" not in e:
                    continue
                xs.append(janela)
                ys.append(e["idf1"]["mean"]); erros.append(e["idf1"]["std"])
                sw.append(e["id_switches"]["mean"]); sw_err.append(e["id_switches"]["std"])
            if not xs:
                continue
            a.errorbar(xs, ys, yerr=erros, marker="o", capsize=4, lw=2,
                       color=cores.get(celula), label=celula)
            b.errorbar(xs, sw, yerr=sw_err, marker="s", capsize=4, lw=2,
                       color=cores.get(celula), label=celula)

        for ax, titulo, rotulo in (
            (a, "IDF1 por janela de BPTT", "IDF1 (validação)"),
            (b, "ID switches por janela de BPTT", "ID switches"),
        ):
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(self.janelas))
            ax.set_xticklabels([str(j) for j in self.janelas])
            ax.set_xlabel("T — comprimento da janela de BPTT truncado")
            ax.set_ylabel(rotulo)
            ax.set_title(titulo, fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)

        fig.suptitle("Eixo 1 — mesma quantidade de parâmetros, 3 seeds, média ± desvio",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(self.run_dir / "eixo1.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
