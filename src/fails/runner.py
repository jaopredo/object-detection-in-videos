"""Parte 4 — galeria de falhas e horizonte de memória.

O enunciado pede três coisas, e as três saem daqui:

1. **três trechos** em que o modelo final erra feio, cada um com a figura (tira de quadros
   com gabarito e predição coloridos por identidade, mais o mapa intermediário — no nosso
   caso a caixa que a recorrência previu) e um **diagnóstico** escrito. Além da figura
   (exigida), cada trecho também sai como vídeo (``p4_falha_N.mp4``) — o buraco inteiro
   quadro a quadro, não só as 5 amostras da tira, para ver a identidade se perder ao vivo;
2. o **horizonte de memória efetivo**, medido analiticamente (norma do gradiente) e
   empiricamente (sobrevivência à oclusão), comparado com a distribuição de duração de
   oclusão do dataset;
3. **uma correção** — escolher um diagnóstico, implementar a mudança que ele sugere e
   mostrar o antes/depois.

A correção está em ``configs/mot17_gru_dtaug.yaml`` e o antes/depois sai de
``--mode stress`` nos dois checkpoints; este módulo produz (1) e (2).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.memory import (
    horizonte_analitico, horizonte_empirico, norma_do_gradiente_por_distancia,
    sobrevivencia_a_oclusao, taxa_por_duracao,
)
from src.core.config import AppConfig
from src.data.mot17 import ler_quadro
from src.data.pipeline import DataPipeline
from src.evaluation.occlusion import occlusion_runs
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.tracking.rnn_tracker import RNNTracker


def cor(track_id: int):
    """Cor estável por identidade — a mesma convenção da figura da Parte 0 e do notebook."""
    return plt.cm.tab20((int(track_id) * 7) % 20)


class FailsRunner:
    """Gera a galeria de falhas e as duas medidas do horizonte de memória."""

    def __init__(self, app_config: AppConfig, checkpoint: Path | None = None,
                 split: str | None = None, n_trechos: int = 3):
        self.app_config = app_config
        self.cfg = app_config.get_eval_config()
        self.split = split or self.cfg.split
        self.checkpoint = Path(checkpoint) if checkpoint else Path(self.cfg.output_dir) / "best.pth"
        self.n_trechos = n_trechos
        self.botoes = {k: v for k, v in self.cfg.tracking.items() if k != "tracker"}
        self.raiz = self.cfg.data.get("root", "src/data/datasets/MOT17")

    @torch.no_grad()
    def _rastrear(self, model, dataset):
        saida = []
        for i in range(len(dataset)):
            gt = dataset[i]
            dets = dataset.detections(i)
            pred = RNNTracker(model, **self.botoes).run(
                dets, gt.name, gt.fps, gt.width, gt.height)
            saida.append((gt, dets, pred))
        return saida

    def run(self) -> dict:
        model = ModelFactoryRegistry.build(self.app_config.get_train_config())
        load_checkpoint(self.checkpoint, model, "cpu")
        model.eval()

        dataset = DataPipeline(self.app_config, config=self.cfg).build_split(self.split)
        rastreado = self._rastrear(model, dataset)

        # irmã de `output_dir` (ex.: outputs/p2 → outputs/p4), não filha: a Parte 4 reaproveita
        # o config da Parte 2 só para achar o checkpoint, o que ela produz é outro artefato.
        out = Path(self.cfg.output_dir).parent / "p4"
        out.mkdir(parents=True, exist_ok=True)
        figuras = Path(self.cfg.figures_dir)

        memoria = self._horizonte(model, rastreado, figuras)
        memoria["por_celula"] = self._comparar_celulas()
        galeria = self._galeria(rastreado, memoria, figuras, out)

        (out / "p4.json").write_text(
            json.dumps({"memoria": memoria, "galeria": galeria}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        print(f"\n  detalhe: {out / 'p4.json'}")
        return {"memoria": memoria, "galeria": galeria}

    # ------------------------------------------------------- horizonte de memória

    def _horizonte(self, model, rastreado, figuras: Path) -> dict:
        _, val = DataPipeline(self.app_config).build_dataloaders()
        normas = norma_do_gradiente_por_distancia(model, next(iter(val)))
        h_analitico = horizonte_analitico(normas)

        eventos = [e for gt, _, pred in rastreado
                   for e in sobrevivencia_a_oclusao(gt, pred)]
        h_empirico = horizonte_empirico(eventos)
        duracoes = [r.duration for gt, _, _ in rastreado for r in occlusion_runs(gt)]

        resumo = {
            "horizonte_analitico": h_analitico,
            "horizonte_empirico": h_empirico,
            "queda_do_gradiente": float(normas[0] / normas[min(h_analitico, len(normas) - 1)])
                                  if len(normas) > 1 else 1.0,
            "normas": normas.tolist(),
            "n_oclusoes_no_dataset": len(duracoes),
            "oclusao_media": float(np.mean(duracoes)) if duracoes else 0.0,
            "oclusao_p90": float(np.percentile(duracoes, 90)) if duracoes else 0.0,
            "oclusoes_rastreadas": len(eventos),
            "taxa_de_sobrevivencia": float(np.mean([e["sobreviveu"] for e in eventos]))
                                     if eventos else 0.0,
            "por_duracao": taxa_por_duracao(eventos),
            "fracao_de_oclusoes_alem_do_horizonte": float(
                np.mean([d > h_analitico for d in duracoes])) if duracoes else 0.0,
        }

        print("\n  HORIZONTE DE MEMÓRIA")
        print(f"    analítico  {h_analitico} passos "
              f"(o gradiente cai {resumo['queda_do_gradiente']:.0f}x até lá)")
        print(f"    empírico   {h_empirico} quadros "
              f"(sobrevivência acima de 50%)")
        print(f"    a régua    oclusão média {resumo['oclusao_media']:.1f} quadros, "
              f"p90 {resumo['oclusao_p90']:.0f}")
        print(f"    {100 * resumo['fracao_de_oclusoes_alem_do_horizonte']:.0f}% das oclusões "
              f"do dataset são mais longas que o horizonte analítico")

        self._figura_horizonte(resumo, duracoes, figuras / f"p4_horizonte_{self.split}.png")
        print(f"    figura: {figuras / f'p4_horizonte_{self.split}.png'}")
        return resumo

    def _figura_horizonte(self, resumo, duracoes, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))

        normas = np.array(resumo["normas"])
        k = np.arange(len(normas))
        a.semilogy(k, normas / normas[0], "o-", color="#C44E52", lw=2)
        a.axhline(0.01, color="gray", ls="--", lw=1, label="1% — corte do horizonte")
        a.axvline(resumo["horizonte_analitico"], color="#4C72B0", ls=":", lw=2,
                  label=f"horizonte analítico = {resumo['horizonte_analitico']}")
        a.set_xlabel("k — passos para trás")
        a.set_ylabel(r"$\|\partial L_t / \partial h_{t-k}\|$  (relativo a $k=0$)")
        a.set_title("Medida analítica: até onde o gradiente chega", fontsize=10)
        a.grid(alpha=0.25, which="both")
        a.legend(fontsize=8)

        if duracoes:
            b.hist(duracoes, bins=range(1, min(max(duracoes), 80) + 2), align="left",
                   color="#CCCCCC", edgecolor="white", label="oclusões do dataset")
        b.set_xlabel("duração da oclusão (quadros)")
        b.set_ylabel("ocorrências")
        eixo = b.twinx()
        faixas = resumo["por_duracao"]
        if faixas:
            centros = [(f["min"] + min(f["max"], 60)) / 2 for f in faixas]
            eixo.plot(centros, [f["taxa"] for f in faixas], "s-", color="#55A868", lw=2,
                      label="identidade sobreviveu")
            for f, c in zip(faixas, centros):
                eixo.annotate(f"n={f['n']}", xy=(c, f["taxa"]), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=7,
                              color="#55A868")
        eixo.axvline(resumo["horizonte_analitico"], color="#4C72B0", ls=":", lw=2)
        eixo.set_ylim(-0.05, 1.08)
        eixo.set_ylabel("taxa de sobrevivência")
        b.set_title("Medida empírica, contra a régua do dataset", fontsize=10)
        b.set_xlim(0, min(max(duracoes) if duracoes else 40, 80))

        linhas = b.get_legend_handles_labels()[0] + eixo.get_legend_handles_labels()[0]
        rotulos = b.get_legend_handles_labels()[1] + eixo.get_legend_handles_labels()[1]
        b.legend(linhas, rotulos, fontsize=8, loc="upper right")

        fig.suptitle(
            f"Horizonte de memória: o gradiente morre em {resumo['horizonte_analitico']} passos, "
            f"e {100 * resumo['fracao_de_oclusoes_alem_do_horizonte']:.0f}% das oclusões "
            f"duram mais que isso",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def _comparar_celulas(self) -> dict | None:
        """A curva de gradiente das três células, se a ablação da Parte 3 já rodou.

        O enunciado manda fazer esta comparação e diz que "os checkpoints já existem". Se
        ainda não existirem, a Parte 4 continua funcionando sem ela em vez de quebrar — as
        duas medidas do horizonte não dependem da ablação.
        """
        from src.analysis.comparar_celulas import comparar, ultima_ablacao

        try:
            ultima_ablacao()
        except FileNotFoundError:
            print("\n  (ablação da Parte 3 ainda não rodou — comparação entre células pulada)")
            return None

        janela = self.app_config.get_train_config().train.get("window", 16)
        try:
            return comparar(self.app_config.config_path, janela=janela)
        except (RuntimeError, FileNotFoundError) as erro:
            print(f"\n  (comparação entre células pulada: {erro})")
            return None

    # ------------------------------------------------------------------- galeria

    def _galeria(self, rastreado, memoria, figuras: Path, out: Path) -> list[dict]:
        """As três falhas, escolhidas para serem **diferentes** umas das outras.

        A primeira versão pegava simplesmente as três oclusões mais longas. Saíram três
        falhas da mesma sequência, com 205, 127 e 115 quadros de buraco, e três diagnósticos
        idênticos — uma galeria que mostra o mesmo caso três vezes não é uma galeria.

        Aqui as candidatas são agrupadas por faixa de duração (curta, média, longa) e se
        tira a pior de cada faixa, preferindo sequências diferentes. Uma falha de 8 quadros
        e uma de 200 têm causas diferentes, e é isso que a Parte 4 quer expor.
        """
        candidatos = []
        for gt, _, pred in rastreado:
            for e in sobrevivencia_a_oclusao(gt, pred):
                if not e["sobreviveu"]:
                    candidatos.append({**e, "sequencia": gt.name, "gt": gt, "pred": pred})

        faixas = [(1, 10), (11, 40), (41, 10 ** 6)]
        escolhidos, sequencias_usadas = [], set()
        for menor, maior in faixas:
            na_faixa = [c for c in candidatos if menor <= c["duracao"] <= maior]
            if not na_faixa:
                continue
            # prefere uma sequência ainda não usada; se todas já apareceram, pega a mais longa
            inedito = [c for c in na_faixa if c["sequencia"] not in sequencias_usadas]
            escolhido = max(inedito or na_faixa, key=lambda c: c["duracao"])
            sequencias_usadas.add(escolhido["sequencia"])
            escolhidos.append(escolhido)

        candidatos = escolhidos + sorted(
            [c for c in candidatos if c not in escolhidos], key=lambda c: -c["duracao"]
        )

        print(f"\n  GALERIA DE FALHAS ({len(candidatos)} oclusões perdidas)")
        galeria = []
        for n, caso in enumerate(candidatos[: self.n_trechos], start=1):
            run = self._run_do_caso(caso)
            diagnostico = self._diagnostico(caso, memoria)
            ok_figura = self._figura_falha(caso, run, diagnostico, figuras / f"p4_falha_{n}.png")
            ok_video = self._video_falha(caso, run, figuras / f"p4_falha_{n}.mp4")
            destino = ("a track morreu" if caso["depois"] is None
                       else f"voltou como {caso['depois']}")
            print(f"    {n}. {caso['sequencia']} id {caso['track_id']} — "
                  f"{caso['duracao']} quadros ocluída, {destino}")
            print(f"       {diagnostico}")
            galeria.append({
                "sequencia": caso["sequencia"], "track_id": caso["track_id"],
                "duracao": caso["duracao"], "antes": caso["antes"],
                "depois": caso["depois"], "diagnostico": diagnostico,
                "figura": str(figuras / f"p4_falha_{n}.png") if ok_figura else None,
                "video": str(figuras / f"p4_falha_{n}.mp4") if ok_video else None,
            })
        return galeria

    def _run_do_caso(self, caso):
        """O ``OcclusionRun`` (início, fim, último/próximo quadro visto) por trás do evento.

        Compartilhado entre a figura e o vídeo: os dois precisam dos mesmos limites do
        buraco, e recalcular a busca duas vezes arriscava os dois discordarem.
        """
        runs = [r for r in occlusion_runs(caso["gt"]) if r.track_id == caso["track_id"]]
        return min(runs, key=lambda r: abs(r.duration - caso["duracao"]))

    def _diagnostico(self, caso, memoria) -> str:
        """O diagnóstico no formato que o enunciado pede, com os números deste caso."""
        h = memoria["horizonte_analitico"]
        queda = memoria["queda_do_gradiente"]
        janela = self.app_config.get_train_config().train.get("window", 16)
        destino = ("a track morreu antes de ele voltar" if caso["depois"] is None
                   else f"voltou com o id {caso['depois']} em vez de {caso['antes']}")
        vezes = caso["duracao"] / max(h, 1)

        if caso["duracao"] <= self.botoes.get("max_age", 30):
            limite = (f"o buraco cabe em max_age={self.botoes.get('max_age')}, então a track "
                      f"estava viva do outro lado — o que faltou foi a previsão continuar "
                      f"apontando para o lugar certo")
        else:
            limite = (f"o buraco é maior que max_age={self.botoes.get('max_age')}, então a "
                      f"track foi morta pela gestão antes de qualquer previsão poder salvá-la")

        return (
            f"esse objeto fica ocluído por {caso['duracao']} quadros — {vezes:.0f}x o "
            f"horizonte analítico de {h} passos, em que a norma do gradiente já caiu "
            f"{queda:.0f}x. Com janela de BPTT de {janela}, o modelo nunca recebeu sinal de "
            f"supervisão que atravessasse esse buraco; {limite}. Resultado: {destino}."
        )

    def _desenhar_quadro(self, ax, t: int, caso: dict, run) -> bool:
        """Um quadro só: gabarito sólido, predição tracejada, recortado em volta do objeto.

        Compartilhado entre a tira estática (``_figura_falha``) e o vídeo (``_video_falha``)
        — é o mesmo desenho; muda só quantos quadros viram arquivo, e se é um PNG ou um MP4.

        O "mapa intermediário relevante" que o enunciado pede é, nesta trilha, **a caixa que
        a recorrência previu**: é a representação interna que decide a associação, e é ela
        que vai derivando durante a oclusão até não casar mais com nada.

        Quando não há gabarito (a maior parte dos quadros do vídeo — é literalmente a
        definição de oclusão) o enquadramento segue a caixa que a recorrência previu, em vez
        de mostrar o quadro inteiro sem recorte. É o que deixa a câmera acompanhando a
        previsão enquanto ela deriva — o próprio fenômeno que esta parte quer mostrar.

        Returns:
            ``False`` se a imagem do quadro não está em disco (``make dados-imagens``).
        """
        gt, pred = caso["gt"], caso["pred"]
        imagem = ler_quadro(self.raiz, gt.name, t)
        if imagem is None:
            return False

        ax.imshow(imagem)
        caixa_gt = gt[t].box_of(caso["track_id"])
        if caixa_gt is not None:
            self._retangulo(ax, caixa_gt, cor(caso["track_id"]), "-", f"gt {caso['track_id']}")

        caixa_pred = None
        for track_id, caixa in zip(pred[t].ids, pred[t].boxes):
            if int(track_id) in (caso["antes"], caso["depois"]):
                self._retangulo(ax, caixa, cor(track_id), "--", f"pred {int(track_id)}")
                caixa_pred = caixa

        # recorta em volta do objeto: o quadro inteiro do MOT17 é 1920x1080 e o pedestre
        # ocupa 40x100 px — sem o recorte, a figura mostra uma multidão e nenhuma caixa
        caixa_referencia = caixa_gt if caixa_gt is not None else caixa_pred
        if caixa_referencia is not None:
            self._enquadrar(ax, caixa_referencia, imagem.shape)

        dentro = run.start <= t <= run.end
        ax.set_title(f"t={t}" + ("  (ocluído)" if dentro else ""), fontsize=9,
                     color="#C44E52" if dentro else "black")
        ax.set_xticks([]); ax.set_yticks([])
        return True

    def _figura_falha(self, caso, run, diagnostico: str, path: Path) -> bool:
        """Tira de quadros: último visto, até 3 amostras dentro do buraco, primeiro visto
        de volta."""
        gt = caso["gt"]
        indices = [run.last_seen,
                   *np.linspace(run.start, run.end, num=min(3, run.duration), dtype=int).tolist(),
                   run.next_seen]
        indices = sorted(set(int(i) for i in indices))

        fig, eixos = plt.subplots(1, len(indices), figsize=(3.0 * len(indices), 3.6))
        eixos = np.atleast_1d(eixos)

        for ax, t in zip(eixos, indices):
            if not self._desenhar_quadro(ax, t, caso, run):
                plt.close(fig)
                print(f"       (sem imagem em disco para {gt.name} — figura pulada)")
                return False

        fig.suptitle(
            f"{gt.name} — identidade {caso['track_id']}, {run.duration} quadros de oclusão\n"
            + diagnostico,
            fontsize=9.5,
        )
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return True

    def _video_falha(self, caso, run, path: Path, margem: int = 5) -> bool:
        """O trecho inteiro como vídeo, quadro a quadro, em vez de só 5 amostras.

        Mesmo desenho da figura estática (``_desenhar_quadro``); cada quadro vira um array
        RGB via o canvas do matplotlib, e a sequência de arrays vira MP4 com
        ``imageio.mimwrite`` — a mesma chamada que ``src/inference/predict.py::gravar_video``
        já usa para o vídeo do ``inferencia.ipynb``, então não é dependência nova.

        ``margem`` quadros de folga antes do sumiço e depois do reaparecimento dão contexto
        de como a identidade era rastreada normalmente antes e depois do buraco.
        """
        import imageio.v2 as imageio

        gt = caso["gt"]
        inicio = max(0, run.last_seen - margem)
        fim = min(len(gt) - 1, run.next_seen + margem)

        quadros = []
        for t in range(inicio, fim + 1):
            # figsize/dpi fixos e sem `tight_layout`/`bbox_inches="tight"`: um MP4 exige
            # quadros do mesmo tamanho, e "tight" recorta diferente conforme o texto do
            # título muda de um quadro para o outro (ex.: "(ocluído)" aparece e some)
            fig, ax = plt.subplots(figsize=(5.0, 5.6), dpi=100)
            if not self._desenhar_quadro(ax, t, caso, run):
                plt.close(fig)
                print(f"       (sem imagem em disco para {gt.name} — vídeo pulado)")
                return False
            fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
            fig.canvas.draw()
            quadros.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            plt.close(fig)

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, quadros, fps=gt.fps, macro_block_size=1)
        return True

    @staticmethod
    def _retangulo(ax, caixa, cor_, estilo, rotulo):
        x1, y1, x2, y2 = caixa
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=cor_, lw=2.0, linestyle=estilo))
        ax.annotate(rotulo, xy=(x1, y1), xytext=(0, -4), textcoords="offset points",
                    fontsize=7, color=cor_, weight="bold")

    @staticmethod
    def _enquadrar(ax, caixa, forma, folga: float = 2.5):
        x1, y1, x2, y2 = caixa
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        meia_l = max((x2 - x1) * folga, 90)
        meia_a = max((y2 - y1) * folga / 2, 90)
        ax.set_xlim(max(0, cx - meia_l), min(forma[1], cx + meia_l))
        ax.set_ylim(min(forma[0], cy + meia_a), max(0, cy - meia_a))
