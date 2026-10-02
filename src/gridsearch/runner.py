"""Grid search de hiperparâmetros (lr, batch_size, ...) do treino da Parte 2.

Varre o produto cartesiano de ``grid_search.params`` do YAML (chaves em dotted-path, ex.:
``train.lr``), com ``grid_search.seeds`` seeds por combinação. A validação de cada
combinação **não** é o ``val`` do projeto (MOT17-10): é um holdout interno, ``grid_val``
(MOT17-02), cortado de dentro do ``train`` só para esta busca — ver ``SPLITS`` em
``src/data/mot17.py``. A métrica de seleção é o IDF1 do rastreador completo sobre esse
holdout, calculado uma vez por ``(combinação, seed)`` sobre o ``best.pth`` daquele run — não
a perda de regressão a cada época, que continua decidindo a **época** dentro do run, como
sempre (``train.monitor``).

Depois de escolher a combinação vencedora (maior IDF1 médio entre seeds), um último treino
roda com ela nas quatro sequências de treino inteiras (o ``grid_val``/MOT17-02 volta ao
treino) e validação normal em MOT17-10 — é o modelo de produção, sem nenhuma lógica nova de
parada: é o ``TrainEngine`` de sempre, com o split de sempre.

Esse retreino final grava no **mesmo** ``output_dir``/``figures_dir`` do YAML base (o que
``make treinar`` também usa) — de propósito, e não por acidente: é o que deixa ``--mode
eval``/``fails``/``stress`` reaproveitarem o vencedor sem precisar de ``--checkpoint`` nem de
nenhuma mudança neles. Consequência a assumir: rodar o grid search **sobrescreve** o
checkpoint que já estivesse lá, exatamente como rodar ``make treinar`` de novo sobrescreveria.

Nenhum nome de hiperparâmetro é hardcoded: adicionar uma chave em ``grid_search.params`` no
YAML já entra na grade, sem tocar em código — por isso os overrides são aplicados por
dotted-path (``_set_dotted``), e não por uma lista fixa de campos conhecidos (diferente da
ablação da Parte 3, que varia só célula e janela).
"""

import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

from src.core.grid_stats import mean_std_metrics
from src.core.standalone_config import StandaloneConfig

#: métricas do tracker reportadas por combinação, mesmo conjunto da ablação da Parte 3
METRICAS = ("idf1", "idp", "idr", "id_switches", "fragmentations", "mota", "val_loss")


def _set_dotted(raw: dict, dotted_key: str, valor) -> None:
    """``_set_dotted(raw, "train.lr", 0.001)`` → ``raw["train"]["lr"] = 0.001``."""
    *partes, folha = dotted_key.split(".")
    alvo = raw
    for p in partes:
        alvo = alvo.setdefault(p, {})
    alvo[folha] = valor


class GridSearchRunner:
    """Varre ``grid_search.params`` × ``grid_search.seeds``, escolhe por IDF1 no holdout
    interno (MOT17-02) e retreina a combinação vencedora com o treino inteiro."""

    def __init__(self, config_base: str, base_output_dir: str = "outputs/gridsearch"):
        self.config_base = Path(config_base)
        base = yaml.safe_load(self.config_base.read_text(encoding="utf-8"))
        grid = base.get("grid_search")
        if not grid or not grid.get("params"):
            raise ValueError(
                f"{self.config_base} não tem bloco `grid_search:` com `params` (dotted-path "
                f"→ lista de valores) e opcionalmente `seeds`."
            )
        self.params: dict[str, list] = grid["params"]
        self.seeds: tuple[int, ...] = tuple(grid.get("seeds", (0,)))
        self.base_output_dir = Path(base_output_dir)

    def run(self) -> dict:
        from src.evaluation.tracking_engine import TrackingEvalEngine
        from src.training.engine import TrainEngine

        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{carimbo}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        nomes = list(self.params)
        combinacoes = list(product(*(self.params[nome] for nome in nomes)))
        total_runs = len(combinacoes) * len(self.seeds)
        print(f"grid search — {len(nomes)} hiperparâmetro(s) × {len(combinacoes)} "
              f"combinação(ões) × {len(self.seeds)} seeds = {total_runs} runs")
        print(f"saída: {self.run_dir}\n")

        resultados: dict = {}
        combos_por_chave: dict[str, dict] = {}
        t0 = time.perf_counter()

        for valores in combinacoes:
            overrides = dict(zip(nomes, valores))
            chave = "_".join(
                f"{nome.split('.')[-1]}{valor}" for nome, valor in overrides.items()
            )
            combos_por_chave[chave] = overrides

            for seed in self.seeds:
                destino = self.run_dir / chave / f"seed_{seed}"
                raw = self._config(overrides, seed, destino, holdout=True)
                app = StandaloneConfig(raw)

                print(f"[{chave} seed={seed}] ... ", end="", flush=True)
                inicio = time.perf_counter()

                TrainEngine(app, resume=False).run_silencioso()
                metricas = TrackingEvalEngine(
                    app, destino / "best.pth", split="grid_val"
                ).run_silencioso()

                dt = time.perf_counter() - inicio
                resultados.setdefault(chave, {})[seed] = metricas
                print(f"IDF1 {metricas['idf1']:.4f} | {dt:.0f}s")

        estatisticas = self._estatisticas(resultados, combos_por_chave)
        vencedora = max(estatisticas, key=lambda c: estatisticas[c]["idf1"]["mean"])
        final = self._retreinar(combos_por_chave[vencedora])

        (self.run_dir / "report.json").write_text(
            json.dumps({
                "params": self.params, "seeds": list(self.seeds),
                "estatisticas": estatisticas, "vencedora": vencedora,
                "overrides_vencedores": combos_por_chave[vencedora], "final": final,
            }, indent=2), encoding="utf-8")
        self._relatorio(estatisticas, vencedora, combos_por_chave, final)

        total = time.perf_counter() - t0
        print(f"\ntempo total: {total / 60:.1f} min")
        print(f"vencedora: {chave_fmt(combos_por_chave[vencedora])}")
        print(f"relatório: {self.run_dir / 'report.md'}")
        print(f"modelo final: {Path(final['output_dir']) / 'best.pth'}")
        return estatisticas

    # ------------------------------------------------------------------------- config

    def _config(self, overrides: dict, seed: int, destino: Path | None, holdout: bool) -> dict:
        """``destino=None`` deixa ``output_dir``/``figures_dir`` como estão no YAML base — é
        o que o retreino final usa, de propósito, para gravar no mesmo lugar que ``make
        treinar``. As runs de busca sempre passam um ``destino`` próprio, para não
        sobrescreverem umas às outras nem o modelo de produção no meio da varredura.
        """
        raw = yaml.safe_load(self.config_base.read_text(encoding="utf-8"))
        raw["seed"] = seed
        for dotted_key, valor in overrides.items():
            _set_dotted(raw, dotted_key, valor)
        if holdout:
            # só durante a busca: valida no holdout interno (MOT17-02), não no `val`
            # oficial do projeto (MOT17-10) — ver docstring do módulo
            raw.setdefault("train", {})["train_split"] = "grid_train"
            raw["train"]["val_split"] = "grid_val"
        if destino is not None:
            raw["output_dir"] = str(destino)
            raw["figures_dir"] = str(destino)
        raw["_origem"] = str(self.config_base)
        return raw

    def _retreinar(self, overrides: dict) -> dict:
        """Hiperparâmetro escolhido, agora com as 4 sequências de treino inteiras (o
        MOT17-02 volta) e validação normal em MOT17-10 — treino padrão, sem split
        customizado, sem lógica nova de parada, gravando no ``output_dir`` do YAML base
        (o mesmo que ``make treinar`` usa — ver docstring do módulo).
        """
        from src.evaluation.tracking_engine import TrackingEvalEngine
        from src.training.engine import TrainEngine

        raw = self._config(overrides, seed=0, destino=None, holdout=False)
        app = StandaloneConfig(raw)
        destino = app.get_train_config().output_dir

        print(f"\nretreino final com {chave_fmt(overrides)} no treino inteiro (4 sequências)...")
        print(f"gravando em {destino} — o mesmo lugar de `make treinar` (sobrescreve o que "
              f"já estiver lá)")
        TrainEngine(app, resume=False).run()
        metricas = TrackingEvalEngine(app, destino / "best.pth", split="val").run_silencioso()
        return {"overrides": overrides, "output_dir": str(destino), **metricas}

    # -------------------------------------------------------------------- estatísticas

    def _estatisticas(self, resultados: dict, combos_por_chave: dict) -> dict:
        saida = {}
        for chave, por_seed in resultados.items():
            entrada = {"overrides": combos_por_chave[chave],
                       "n_params": next(iter(por_seed.values())).get("n_params")}
            entrada.update(mean_std_metrics(por_seed, METRICAS))
            saida[chave] = entrada
        return saida

    def _relatorio(self, estatisticas: dict, vencedora: str, combos_por_chave: dict,
                    final: dict) -> None:
        colunas = list(next(iter(combos_por_chave.values())))
        cabecalho = " | ".join(c.split(".")[-1] for c in colunas)
        linhas = [
            "# Grid search — hiperparâmetros de treino (Parte 2b)",
            "",
            f"Gerado em {datetime.now():%Y-%m-%d %H:%M}. "
            f"{len(self.seeds)} seeds ({', '.join(map(str, self.seeds))}), média ± desvio. "
            f"Avaliado no holdout interno **grid_val** (MOT17-02) — não é o `val` oficial "
            f"do projeto (MOT17-10).",
            "",
            f"| {cabecalho} | IDF1 | val_loss | ID switches |",
            "|" + "---|" * (len(colunas) + 3),
        ]
        for chave in sorted(estatisticas, key=lambda c: -estatisticas[c]["idf1"]["mean"]):
            e = estatisticas[chave]
            fmt = lambda m, c=4: (f"{e[m]['mean']:.{c}f} ± {e[m]['std']:.{c}f}"
                                  if m in e else "—")
            marca = " **← vencedora**" if chave == vencedora else ""
            valores = " | ".join(str(e["overrides"][c]) for c in colunas)
            linhas.append(
                f"| {valores} | {fmt('idf1')} | {fmt('val_loss', 5)} | "
                f"{fmt('id_switches', 1)} |{marca}"
            )

        linhas += [
            "",
            f"**Vencedora**: `{chave_fmt(combos_por_chave[vencedora])}`.",
            "",
            f"**Retreino final** (4 sequências de treino, validação em MOT17-10): "
            f"IDF1 {final['idf1']:.4f} | val_loss {final.get('val_loss', float('nan')):.5f} "
            f"| checkpoint `{Path(final['output_dir']) / 'best.pth'}` "
            f"(mesmo lugar de `make treinar`).",
        ]
        (self.run_dir / "report.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        print("\n" + "\n".join(linhas[4:]))


def chave_fmt(overrides: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in overrides.items())
