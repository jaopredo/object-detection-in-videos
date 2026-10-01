"""Parte 5 — queda de taxa de quadros, **sem retreinar**.

O enunciado pergunta: *"por que um modelo de movimento aprendido em Δt fixo quebra quando Δt
muda? Alimentar Δt na recorrência resolveria?"*

Aqui essa pergunta é **testável**, e não uma especulação para a apresentação, porque Δt é
uma entrada do modelo desde o começo (ver ``src/models/motion_rnn.py``). Rodamos as duas
condições sobre o mesmo modelo e os mesmos vídeos:

**alheio**  o modelo recebe Δt = 1/fps, como se nada tivesse mudado. É o que aconteceria com
    qualquer modelo de movimento que suponha que "um passo" é sempre a mesma coisa.

**informado**  o modelo recebe o Δt de verdade — ``k/fps`` com subamostragem de 1/k.

A diferença entre as duas curvas é a resposta, medida. Se "informado" recuperar a queda, sim;
se não recuperar, a informação sozinha não bastava e o que falta é ter **visto** aquele Δt no
treino — o que também é resultado, e melhor do que um palpite.

O custo é zero: subamostrar é pular linhas das listas que já estão na memória. Nenhum
detector roda de novo, e a fonte de detecções continua exatamente a mesma.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.config import AppConfig
from src.data.detections import Detections
from src.data.pipeline import DataPipeline
from src.data.sequence import Frame, Sequence
from src.metrics.identity import evaluate
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.tracking.rnn_tracker import RNNTracker
from src.tracking import IoUTracker

#: 1 = vídeo original, 2 = metade dos quadros, 5 = um quinto. São os do enunciado.
FATORES = (1, 2, 5)


def subamostrar(sequence: Sequence, k: int) -> Sequence:
    """Fica com um quadro a cada ``k`` e **renumera** os índices a partir de zero.

    Renumerar é o que torna a sequência subamostrada indistinguível de um vídeo gravado
    naquela taxa. Se os índices originais fossem mantidos, o rastreador veria buracos de
    ``k-1`` quadros entre observações e envelheceria as tracks por um motivo que não é o
    do experimento.
    """
    frames = [
        Frame(index=novo, boxes=f.boxes, ids=f.ids, visibility=f.visibility, image=f.image)
        for novo, f in enumerate(sequence.frames[::k])
    ]
    return Sequence(name=f"{sequence.name}@1/{k}", fps=sequence.fps / k,
                    width=sequence.width, height=sequence.height, frames=frames)


def subamostrar_deteccoes(dets: list[Detections], k: int) -> list[Detections]:
    return [Detections(novo, d.boxes, d.scores) for novo, d in enumerate(dets[::k])]


class FramerateStress:
    """Curva de degradação do IDF1 com a queda de taxa de quadros."""

    def __init__(self, app_config: AppConfig, checkpoint: Path | None = None,
                 split: str | None = None, fatores=FATORES):
        self.app_config = app_config
        self.cfg = app_config.get_eval_config()
        self.split = split or self.cfg.split
        self.checkpoint = Path(checkpoint) if checkpoint else Path(self.cfg.output_dir) / "best.pth"
        self.fatores = tuple(fatores)
        self.botoes = {k: v for k, v in self.cfg.tracking.items() if k != "tracker"}

    @torch.no_grad()
    def run(self) -> dict:
        model = ModelFactoryRegistry.build(self.app_config.get_train_config())
        load_checkpoint(self.checkpoint, model, "cpu")
        model.eval()

        dataset = DataPipeline(self.app_config, config=self.cfg).build_split(self.split)
        sequencias = [(dataset[i], dataset.detections(i)) for i in range(len(dataset))]

        condicoes = {
            "IoU (baseline)": lambda k: (IoUTracker(**self.botoes), 1),
            "RNN alheio ao Δt": lambda k: (RNNTracker(model, **self.botoes), 1),
            "RNN informado do Δt": lambda k: (RNNTracker(model, **self.botoes), k),
        }

        resultados = {nome: {} for nome in condicoes}
        print(f"\nParte 5 — queda de taxa de quadros, split {self.split}, sem retreinar\n")
        print(f"  {'condição':<24} " + " ".join(f"{'1/' + str(k):>9}" for k in self.fatores))

        for nome, monta in condicoes.items():
            linha = []
            for k in self.fatores:
                tracker, stride = monta(k)
                idf1, switches = [], 0
                for gt, dets in sequencias:
                    gt_k = subamostrar(gt, k)
                    dets_k = subamostrar_deteccoes(dets, k)
                    # `stride` é o que chega na recorrência como Δt = stride / fps. Na
                    # condição "alheio" ele fica em 1 de propósito: o modelo continua
                    # achando que passou 1/30 s entre observações.
                    pred = tracker.run(dets_k, gt_k.name, gt.fps, gt.width, gt.height,
                                       stride=stride)
                    m = evaluate(gt_k, pred)
                    idf1.append(m.idf1)
                    switches += m.id_switches
                resultados[nome][k] = {"idf1": float(np.mean(idf1)), "id_switches": switches}
                linha.append(f"{np.mean(idf1):>9.4f}")
            print(f"  {nome:<24} " + " ".join(linha))

        self._resposta(resultados)

        out = Path(self.cfg.output_dir) / "p5"
        out.mkdir(parents=True, exist_ok=True)
        (out / "framerate.json").write_text(
            json.dumps({"split": self.split, "fatores": list(self.fatores),
                        "resultados": {n: {str(k): v for k, v in d.items()}
                                       for n, d in resultados.items()}}, indent=2),
            encoding="utf-8")

        figura = Path(self.cfg.figures_dir) / f"p5_framerate_{self.split}.png"
        self._figura(resultados, figura)
        print(f"\n  detalhe: {out / 'framerate.json'}")
        print(f"  figura:  {figura}")
        return resultados

    def _resposta(self, resultados: dict) -> None:
        """A resposta à pergunta do enunciado, em números, no fim da execução."""
        pior = max(self.fatores)
        alheio = resultados["RNN alheio ao Δt"]
        informado = resultados["RNN informado do Δt"]

        queda_alheio = alheio[1]["idf1"] - alheio[pior]["idf1"]
        queda_informado = informado[1]["idf1"] - informado[pior]["idf1"]
        recuperado = informado[pior]["idf1"] - alheio[pior]["idf1"]

        print(f"\n  ALIMENTAR Δt NA RECORRÊNCIA RESOLVE? (a 1/{pior} da taxa original)")
        print(f"    queda sem informar Δt    {queda_alheio:+.4f} de IDF1")
        print(f"    queda informando Δt      {queda_informado:+.4f}")
        print(f"    recuperado pelo Δt       {recuperado:+.4f} "
              f"({100 * recuperado / queda_alheio:+.1f}% da queda)"
              if queda_alheio else "")

    def _figura(self, resultados: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cores = {"IoU (baseline)": "#4C72B0", "RNN alheio ao Δt": "#DD8452",
                 "RNN informado do Δt": "#55A868"}

        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(self.fatores))
        for nome, por_fator in resultados.items():
            a.plot(x, [por_fator[k]["idf1"] for k in self.fatores], "o-", lw=2,
                   color=cores.get(nome), label=nome)
            b.plot(x, [por_fator[k]["id_switches"] for k in self.fatores], "s-", lw=2,
                   color=cores.get(nome), label=nome)

        for ax, rotulo, titulo in (
            (a, "IDF1", "Degradação do IDF1"),
            (b, "ID switches", "ID switches"),
        ):
            ax.set_xticks(x)
            ax.set_xticklabels([f"1/{k}\n({30 / k:.0f} fps)" for k in self.fatores])
            ax.set_xlabel("taxa de quadros, como fração da original")
            ax.set_ylabel(rotulo)
            ax.set_title(titulo, fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
        a.set_ylim(0, None)

        fig.suptitle("Parte 5 — o vídeo subamostrado, sem retreinar o modelo", fontsize=11)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
