"""Grava em disco os vídeos sintéticos e a figura da oclusão — o que ``--mode gen-synth`` faz.

A saída imita a organização do MOTChallenge::

    outputs/p0/sequences/SYNTHT-00/
        img1/000001.jpg ...     quadros numerados a partir de 1
        gt/gt.txt               anotações no formato MOT
        video.mp4               o mesmo conteúdo, para olhar
        seqinfo.ini             metadados (fps, tamanho, nº de quadros)

Não é imitação por estética: é para que o leitor do MOT17 da Parte 1 consiga abrir estas
pastas sem um caminho especial, e para que qualquer script escrito contra o MOT17 funcione
no sintético de graça.
"""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import AppConfig
from src.data.mot_format import write_mot_gt
from src.data.pipeline import DataPipeline
from src.data.sequence import Sequence
from src.evaluation.occlusion import occlusion_runs


class SyntheticRunner:
    """Gera os vídeos dos três splits e a figura obrigatória da Parte 0.

    Args:
        app_config: configuração carregada.
        saida: pasta raiz dos resultados. Padrão: o ``output_dir`` do config.
        salvar_quadros: grava também os PNGs individuais. Desligar deixa a saída bem menor
            quando só o mp4 interessa.
    """

    def __init__(self, app_config: AppConfig, saida: str | Path | None = None,
                 salvar_quadros: bool = True):
        self.app_config = app_config
        self.cfg = app_config.get_train_config()
        self.saida = Path(saida) if saida else self.cfg.output_dir
        self.salvar_quadros = salvar_quadros

    def run(self) -> None:
        splits = DataPipeline(self.app_config).build_datasets()

        todas: list[Sequence] = []
        for nome, dataset in splits._asdict().items():
            print(f"\n{nome}: {len(dataset)} sequências")
            for i in range(len(dataset)):
                seq = dataset[i]
                destino = self._gravar(seq)
                runs = occlusion_runs(seq)
                duracoes = sorted((r.duration for r in runs), reverse=True)
                print(f"  {seq.name:<14} {len(seq):>3} quadros "
                      f"| {len(seq.track_ids):>2} identidades "
                      f"| oclusões {duracoes if duracoes else '—'}")
                todas.append(seq)

        figura = self._figura_oclusao(todas)

        print(f"\nsequências: {self.saida/'sequences'}")
        print(f"figura:     {figura}")

    # ------------------------------------------------------------------ disco

    def _gravar(self, seq: Sequence) -> Path:
        destino = self.saida / "sequences" / seq.name
        destino.mkdir(parents=True, exist_ok=True)

        write_mot_gt(seq, destino / "gt" / "gt.txt")

        (destino / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={seq.name}\n"
            f"imDir=img1\n"
            f"frameRate={seq.fps:g}\n"
            f"seqLength={len(seq)}\n"
            f"imWidth={seq.width}\n"
            f"imHeight={seq.height}\n"
            "imExt=.png\n",
            encoding="utf-8",
        )

        if seq.has_images():
            if self.salvar_quadros:
                pasta = destino / "img1"
                pasta.mkdir(exist_ok=True)
                for frame in seq:
                    imageio.imwrite(pasta / f"{frame.index + 1:06d}.png", frame.image)
            imageio.mimwrite(
                destino / "video.mp4",
                [f.image for f in seq],
                fps=seq.fps,
                macro_block_size=1,  # 128x128 não é múltiplo de 16; sem isso o ffmpeg redimensiona
            )
        return destino

    # ------------------------------------------------------------------ figura

    def _figura_oclusao(self, sequencias: list[Sequence]) -> Path:
        """A figura que o enunciado exige: uma trajetória que some por N quadros e volta.

        Escolhe, entre todas as sequências geradas, a oclusão mais longa — é a que mostra o
        fenômeno com mais clareza. Dois painéis:

        **em cima**, a tira de quadros antes / durante / depois, com a caixa amodal
        desenhada na cor da identidade. Durante a oclusão a caixa continua lá e o objeto
        não: é exatamente isso que o modelo temporal vai ter que aprender a sustentar.

        **embaixo**, a visibilidade quadro a quadro com o vão sombreado e anotado. É a
        prova de que a oclusão é real e de que a duração é a que o config pediu — não uma
        impressão de quem olhou o vídeo.
        """
        escolhido = self._melhor_exemplo(sequencias)

        path = Path(self.cfg.figures_dir) / "oclusao.png"
        path.parent.mkdir(parents=True, exist_ok=True)

        if escolhido is None:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "nenhuma oclusão gerada — aumente occlusion_duration",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            plt.savefig(path, dpi=120)
            plt.close(fig)
            return path

        seq, run = escolhido
        cor = self._cor_da_identidade(run.track_id)

        serie = seq.visibility_of(run.track_id)

        # quadros mostrados: um antes em que o objeto aparece de fato, três durante a
        # oclusão, e um depois. Mais que isso vira uma tira ilegível.
        # Usar last_seen/next_seen mostraria o objeto já quase engolido pelo poste —
        # tecnicamente "visível", visualmente nada.
        antes = self._quadro_bem_visivel(serie, range(run.start - 1, -1, -1), run.last_seen)
        depois = self._quadro_bem_visivel(serie, range(run.end + 1, len(seq)), run.next_seen)
        durante = np.linspace(run.start, run.end, num=min(3, run.duration), dtype=int)
        indices = [antes, *durante.tolist(), depois]

        fig = plt.figure(figsize=(2.1 * len(indices), 5.2))
        grade = fig.add_gridspec(2, len(indices), height_ratios=[2.1, 1.0], hspace=0.28)

        for coluna, t in enumerate(indices):
            ax = fig.add_subplot(grade[0, coluna])
            frame = seq[t]
            ax.imshow(frame.image)

            box = frame.box_of(run.track_id)
            if box is not None:
                x1, y1, x2, y2 = box
                vis = frame.visibility[np.flatnonzero(frame.ids == run.track_id)[0]]
                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor=cor, lw=2.0,
                    linestyle="-" if vis > 0 else "--",
                ))
                ax.set_title(f"t={t}  vis={vis:.2f}", fontsize=8,
                             color="black" if vis > 0 else "#C44E52")
            ax.axis("off")

        ax = fig.add_subplot(grade[1, :])
        ax.plot(serie, color=cor, lw=1.8)
        ax.axvspan(run.start - 0.5, run.end + 0.5, color="#C44E52", alpha=0.15)
        ax.annotate(
            f"ocluído por {run.duration} quadros",
            xy=((run.start + run.end) / 2, 0.5),
            ha="center", va="center", fontsize=9, color="#C44E52",
        )
        for t in indices:
            ax.axvline(t, color="gray", lw=0.6, alpha=0.5)
        ax.set_xlim(0, len(seq) - 1)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("quadro")
        ax.set_ylabel("visibilidade")

        fig.suptitle(
            f"{seq.name} — identidade {run.track_id} some no quadro {run.start} "
            f"e volta no {run.next_seen}, com o mesmo id",
            fontsize=11,
        )
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _quadro_bem_visivel(serie, candidatos, padrao: int, minima: float = 0.6) -> int:
        """Primeiro quadro de ``candidatos`` em que o objeto aparece de fato."""
        for t in candidatos:
            if serie[t] >= minima:
                return t
        return padrao

    @staticmethod
    def _melhor_exemplo(sequencias: list[Sequence], minima: float = 0.6):
        """Escolhe a oclusão que melhor *ilustra* o fenômeno, não a mais longa.

        A oclusão mais longa costuma ser degenerada: um objeto que nasce já quase escondido
        atrás de outro, fica invisível quase o vídeo inteiro e reaparece de raspão. A
        duração é grande e a figura não mostra nada — não dá para ver o objeto antes nem
        depois, então o "some e volta" não se lê.

        O critério aqui é: entre as oclusões em que o objeto está **claramente visível**
        dos dois lados (pelo menos ``minima``), pega a mais longa. Se nenhuma qualificar,
        cai para a mais longa de todas, para a figura existir de qualquer jeito.

        Returns:
            Par (sequência, trecho), ou ``None`` se não houver oclusão nenhuma.
        """
        todos = [(s, r) for s in sequencias for r in occlusion_runs(s)]
        if not todos:
            return None

        def bem_visivel(seq: Sequence, run) -> bool:
            # o pico antes e depois do vão, não o valor em last_seen/next_seen: ao entrar
            # no poste a visibilidade cai gradualmente, então o último quadro com
            # visibilidade não-nula mostra uma nesga de 3% e não serve de critério
            serie = seq.visibility_of(run.track_id)
            antes, depois = serie[:run.start], serie[run.end + 1:]
            return (len(antes) and antes.max() >= minima
                    and len(depois) and depois.max() >= minima)

        bons = [(s, r) for s, r in todos if bem_visivel(s, r)]
        return max(bons or todos, key=lambda par: par[1].duration)

    @staticmethod
    def _cor_da_identidade(track_id: int):
        """Cor estável por identidade — a mesma convenção do notebook de inferência."""
        return plt.cm.tab20(track_id % 20)
