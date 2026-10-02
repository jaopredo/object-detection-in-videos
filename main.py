"""Entry point único do projeto.

Uso:
    python -m main --mode gen-synth --synthetic   # Parte 0: gera os vídeos e a figura
    python -m main --mode eval --synthetic        # caracteriza um split (oclusão, densidade)
    python -m main --config path                  # config YAML customizado

Os modos ``train`` e ``both`` existem mas ainda não têm o que rodar: nada está registrado
em ``ModelFactoryRegistry``, porque as partes 0 e 1 do assignment funcionam com a fonte de
detecções congelada e sem treino nenhum. Eles falham com uma mensagem que diz isso.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig


def parse_arguments() -> argparse.Namespace:
    """Parse automático dos argumentos inline.

    Returns:
        argparse.Namespace: namespace com os argumentos e seus valores.
    """
    parser = argparse.ArgumentParser(description="PA2 — identidade ao longo do tempo")
    parser.add_argument(
        "--mode",
        choices=["gen-synth", "sweep", "baseline", "train", "eval", "both", "ablation",
                 "grid-search", "stress", "fails"],
        default="gen-synth",
        help="gen-synth: gera os vídeos sintéticos da Parte 0; sweep: gira os botões do "
             "gerador e mede onde o baseline quebra (Parte 0); baseline: rastreamento "
             "ingênuo por IoU (Parte 1); eval: caracteriza um split; train/both: treino do "
             "modelo temporal (Parte 2); ablation: Eixo 1 da Parte 3; grid-search: busca de "
             "lr/batch_size com holdout interno (MOT17-02) e retreino final",
    )
    parser.add_argument(
        "--config", default="configs/synthetic.yaml",
        help="caminho do YAML de configuração (padrão: configs/synthetic.yaml)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="atalho para --config configs/synthetic.yaml",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "trainval"], default=None,
        help="--mode eval: em qual conjunto avaliar. O padrão vem do config (val). "
             "Usar 'test' é uma decisão explícita: ele só deve ser tocado no fim.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="retoma o treino de <output_dir>/last.pth, se existir",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="caminho do checkpoint para avaliação (padrão: <output_dir>/best.pth)",
    )
    parser.add_argument(
        "--gen-output", default=None,
        help="--mode gen-synth: pasta de saída (padrão: o output_dir do config)",
    )
    parser.add_argument(
        "--no-frames", action="store_true",
        help="--mode gen-synth: grava só o mp4, sem os PNGs de cada quadro",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.synthetic:
        args.config = "configs/synthetic.yaml"

    app_config = AppConfig(args.config)

    # ===== PARTE 0 — GERAÇÃO DOS VÍDEOS SINTÉTICOS =====
    if args.mode == "gen-synth":
        from src.synthetic.runner import SyntheticRunner
        SyntheticRunner(
            app_config, saida=args.gen_output, salvar_quadros=not args.no_frames
        ).run()
        return

    # ===== PARTE 0 — VARREDURA DOS BOTÕES =====
    if args.mode == "sweep":
        from src.synthetic.sweep import SweepRunner
        SweepRunner(app_config, saida=args.gen_output).run()
        return

    # ===== PARTE 1 — BASELINE POR QUADRO =====
    if args.mode == "baseline":
        from src.baseline.runner import BaselineRunner
        BaselineRunner(app_config, split=args.split).run()
        return

    # ===== PARTE 3 — ABLAÇÃO (EIXO 1) =====
    if args.mode == "ablation":
        from src.ablation.runner import AblationRunner
        AblationRunner(args.config).run()
        return

    # ===== PARTE 2b — GRID SEARCH DE HIPERPARÂMETROS =====
    if args.mode == "grid-search":
        from src.gridsearch.runner import GridSearchRunner
        GridSearchRunner(args.config).run()
        return

    # ===== PARTE 4 — GALERIA DE FALHAS E HORIZONTE DE MEMÓRIA =====
    if args.mode == "fails":
        from src.fails.runner import FailsRunner
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        FailsRunner(app_config, checkpoint, split=args.split).run()
        return

    # ===== PARTE 5 — TESTE DE ESTRESSE =====
    if args.mode == "stress":
        from src.stress.framerate import FramerateStress
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        FramerateStress(app_config, checkpoint, split=args.split).run()
        return

    # ===== TREINO (Parte 2) =====
    # importado aqui dentro de propósito: o engine carrega o torch, e as partes 0 e 1 não
    # precisam dele. Importar no topo obrigaria a instalar torch para gerar um vídeo.
    if args.mode in ("train", "both"):
        from src.training.engine import TrainEngine
        Path(app_config.get_train_config().output_dir).mkdir(parents=True, exist_ok=True)
        TrainEngine(app_config, resume=args.resume).run()

    # ===== AVALIAÇÃO =====
    if args.mode in ("eval", "both"):
        cfg = app_config.get_eval_config()
        if args.split:
            cfg.split = args.split
        checkpoint = Path(args.checkpoint) if args.checkpoint else None

        # Com modelo declarado, avaliar é comparar rastreadores (Parte 2 em diante). Sem
        # modelo, é caracterizar o conjunto — densidade, identidades, oclusão —, que é o que
        # as Partes 0 e 1 precisam e o que o EvalEngine antigo faz.
        if cfg.model.get("name"):
            from src.evaluation.tracking_engine import TrackingEvalEngine
            TrackingEvalEngine(app_config, checkpoint, split=cfg.split).run()
        else:
            from src.evaluation.engine import EvalEngine
            EvalEngine(app_config, checkpoint).run()


if __name__ == "__main__":
    main()
