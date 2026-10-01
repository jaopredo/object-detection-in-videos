"""A curva de gradiente da RNN simples contra a do modelo com portas, na mesma janela.

O enunciado é explícito na Parte 4: *"Se fizeram o Eixo 1 da Parte 3, comparem a curva da
RNN simples com a do modelo com portas, na mesma janela — os checkpoints já existem."* Eles
existem, e é isto que os lê.

A pergunta é a da aula de RNN: **as portas existem por causa do gradiente que some.** Se a
teoria vale no nosso modelo, a curva ``‖∂L_t/∂h_{t−k}‖`` da RNN simples tem que cair mais
rápido que a da LSTM e a da GRU — e o horizonte de memória da RNN simples tem que ser menor.

A mesma janela nas três é indispensável: comparar uma RNN treinada com T=4 contra uma GRU
com T=32 mediria a janela, não a célula.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.memory import horizonte_analitico, norma_do_gradiente_por_distancia
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry

CORES = {"rnn": "#C44E52", "lstm": "#4C72B0", "gru": "#55A868"}
ROTULOS = {"rnn": "RNN simples", "lstm": "LSTM", "gru": "GRU"}


def ultima_ablacao(base: str | Path = "outputs/ablation") -> Path:
    """A run de ablação mais recente."""
    runs = sorted(Path(base).glob("run_*"))
    if not runs:
        raise FileNotFoundError(
            f"nenhuma ablação em {base}. Rode `make parte3` antes — a comparação de "
            f"gradiente da Parte 4 usa os checkpoints do Eixo 1."
        )
    return runs[-1]


def comparar(
    config_base: str | Path = "configs/mot17_gru.yaml",
    janela: int = 16,
    seed: int = 0,
    run: Path | None = None,
    destino: Path | None = None,
) -> dict:
    """Compara as três células na mesma janela e grava a figura e o JSON.

    Args:
        janela: qual ``T`` usar. Tem que ser o mesmo nas três.
        seed: qual das 3 seeds. A curva de gradiente é uma propriedade da arquitetura
            treinada, e as seeds servem para dizer se a diferença entre células é maior que
            a variação entre inicializações — ver ``dispersao_entre_seeds``.
    """
    from src.ablation.runner import StandaloneConfig
    from src.data.pipeline import DataPipeline

    run = run or ultima_ablacao()
    raw = yaml.safe_load(Path(config_base).read_text(encoding="utf-8"))
    raw["train"]["window"] = janela
    raw["train"]["stride"] = max(1, janela // 2)

    resultados = {}
    for celula in ("rnn", "lstm", "gru"):
        checkpoint = run / f"{celula}_T{janela}" / f"seed_{seed}" / "best.pth"
        if not checkpoint.exists():
            print(f"  {celula}_T{janela}/seed_{seed}: sem checkpoint, pulando")
            continue

        cfg_celula = {**raw, "model": {**raw["model"], "cell": celula}}
        cfg_celula["model"].pop("hidden", None)
        app = StandaloneConfig(cfg_celula)

        model = ModelFactoryRegistry.build(app.get_train_config())
        load_checkpoint(checkpoint, model, "cpu")

        _, val = DataPipeline(app).build_dataloaders()
        normas = norma_do_gradiente_por_distancia(model, next(iter(val)))
        resultados[celula] = {
            "normas": normas.tolist(),
            "horizonte": horizonte_analitico(normas),
            "n_params": model.n_parametros(),
            # quantas vezes o gradiente encolhe a cada passo, na média geométrica: é o número
            # que resume a curva num só, e é comparável entre células
            "decaimento_por_passo": float(
                (normas[0] / max(normas[-1], 1e-30)) ** (1 / max(len(normas) - 1, 1))
            ),
        }

    if not resultados:
        raise RuntimeError(f"nenhum checkpoint de T={janela} em {run}")

    destino = Path(destino) if destino else Path("outputs/figures")
    destino.mkdir(parents=True, exist_ok=True)
    _figura(resultados, janela, destino / f"p4_gradiente_celulas_T{janela}.png")
    (Path("outputs/p2/p4") / f"gradiente_celulas_T{janela}.json").parent.mkdir(
        parents=True, exist_ok=True)
    (Path("outputs/p2/p4") / f"gradiente_celulas_T{janela}.json").write_text(
        json.dumps({"janela": janela, "seed": seed, "run": str(run),
                    "resultados": resultados}, indent=2), encoding="utf-8")

    print(f"\n  GRADIENTE POR CÉLULA (T={janela}, seed {seed})")
    print(f"    {'célula':<12} {'parâmetros':>11} {'horizonte':>10} {'decaimento/passo':>18}")
    for celula, r in resultados.items():
        print(f"    {ROTULOS[celula]:<12} {r['n_params']:>11,} {r['horizonte']:>10} "
              f"{r['decaimento_por_passo']:>17.2f}x")
    print(f"    figura: {destino / f'p4_gradiente_celulas_T{janela}.png'}")
    return resultados


def _figura(resultados: dict, janela: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    for celula, r in resultados.items():
        normas = np.array(r["normas"])
        ax.semilogy(np.arange(len(normas)), normas / normas[0], "o-", lw=2,
                    color=CORES[celula],
                    label=f"{ROTULOS[celula]} — horizonte {r['horizonte']}, "
                          f"{r['decaimento_por_passo']:.1f}x por passo")

    ax.axhline(0.01, color="gray", ls="--", lw=1)
    ax.annotate("1% — corte do horizonte", xy=(0.02, 0.011), xycoords=("axes fraction", "data"),
                fontsize=8, color="gray")
    ax.set_xlabel("k — passos para trás")
    ax.set_ylabel(r"$\|\partial L_t / \partial h_{t-k}\|$  (relativo a $k=0$)")
    ax.set_title(
        f"Parte 4 — o gradiente que some, por célula (mesma janela T={janela},\n"
        f"mesmo orçamento de parâmetros, mesma seed)",
        fontsize=10,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
