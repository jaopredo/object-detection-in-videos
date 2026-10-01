"""Motor de treino: loop, validação e checkpoint.

Herdado do PA1 (``semantic-segmentation/src/training/engine.py``). O que é genérico —
fixação de seed, gravação e retomada de checkpoint, histórico por época, separação entre
``best.pth`` e ``last.pth`` — veio inteiro, porque não tem nada de segmentação nisso.

O que mudou:

**A seleção de época passou a ser configurável.** No PA1 o melhor modelo era sempre o de
maior IoU de validação, com o nome da métrica escrito no código. Aqui o config declara
``train.monitor`` (qual métrica) e ``train.monitor_mode`` (``max`` ou ``min``), porque a
Parte 2 vai querer selecionar por IDF1 e a ablação da Parte 3 pode querer por perda.

**A amostra deixou de ser uma imagem.** O que entra no lote são janelas de trajetória
recortadas das sequências — ver ``DataPipeline.build_dataloaders``, que só existe a partir
da Parte 2.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.core.config import AppConfig
from src.data.pipeline import DataPipeline
from src.losses.factory import LossFactoryRegistry
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry


class TrainEngine:
    def __init__(self, app_config: AppConfig, resume: bool = False):
        self.app_config = app_config
        self.cfg = app_config.get_train_config()
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t = self.cfg.train
        #: nome da métrica de validação que decide qual época vira best.pth
        self.monitor = t.get("monitor", "val_loss")
        self.monitor_mode = t.get("monitor_mode", "min")
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(
                f"train.monitor_mode deve ser 'min' ou 'max', não {self.monitor_mode!r}"
            )

    def _melhor(self, novo: float, atual: float) -> bool:
        return novo > atual if self.monitor_mode == "max" else novo < atual

    def _pior_possivel(self) -> float:
        return -float("inf") if self.monitor_mode == "max" else float("inf")

    def _fixed_seeds(self):
        """Ajusta a seed aleatória para manter os resultados reprodutíveis.

        A seed sozinha não basta em GPU: com ``cudnn.benchmark`` ligado, o cuDNN escolhe o
        algoritmo de convolução medindo tempo, e a escolha pode mudar de uma execução para
        outra; alguns desses algoritmos também somam em ordem variável. As duas flags fixam
        algoritmos determinísticos. Isso garante repetir o resultado **na mesma máquina** —
        GPU, driver e versão do PyTorch diferentes ainda dão números diferentes.
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _save_checkpoint(self, path, model, optimizer, epoch, melhor, history):
        """Salva o estado COMPLETO do treino.

        Só os pesos não bastam para retomar: o Adam mantém médias móveis dos gradientes
        (os momentos), e recomeçar sem elas faz as primeiras épocas depois do resume
        saírem instáveis. O histórico vai junto para o log não perder as épocas antigas.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "monitor": self.monitor,
            "best_metric": melhor,
            "history": history,
            "config": {**self.cfg.__dict__, "output_dir": str(self.cfg.output_dir)},
        }, path)

    def _load_checkpoint(self, path, model, optimizer):
        """Restaura o estado de um treino interrompido.

        Returns:
            Tupla (primeira época a rodar, melhor métrica até aqui, histórico).
        """
        estado = load_checkpoint(path, model, self.device)
        if "optimizer" in estado:
            optimizer.load_state_dict(estado["optimizer"])
        epoca = estado.get("epoch", 0)
        melhor = estado.get("best_metric", self._pior_possivel())
        print(f"retomando de {path} — época {epoca}, melhor {self.monitor} {melhor:.4f}\n")
        return epoca + 1, melhor, estado.get("history", [])

    def run(self) -> None:
        cfg = self.cfg
        t = cfg.train

        self._fixed_seeds()

        train_loader, val_loader = DataPipeline(self.app_config).build_dataloaders()

        model = ModelFactoryRegistry.build(cfg).to(self.device)
        self.loss = LossFactoryRegistry.build(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

        n_params = sum(p.numel() for p in model.parameters())
        print(f"modelo: {type(model).__name__} | dispositivo: {self.device} "
              f"| parâmetros: {n_params/1e3:.1f}k")
        print(f"treino: {len(train_loader.dataset)} janelas "
              f"| validação: {len(val_loader.dataset)} janelas\n")

        # =================== LOOP DE TREINO ===================
        history, melhor = [], self._pior_possivel()
        primeira_epoca = 1

        # `last.pth` guarda o estado completo e é gravado a cada `save_every` épocas — a
        # rede de segurança para queda de energia, sessão do Colab que expira, ou as 3
        # seeds da ablação da Parte 3.
        ultimo = cfg.output_dir / "last.pth"
        if self.resume and ultimo.exists():
            primeira_epoca, melhor, history = self._load_checkpoint(ultimo, model, optimizer)
        elif self.resume:
            print(f"--resume pedido, mas {ultimo} não existe: começando do zero\n")

        save_every = t.get("save_every", 5)
        t_start = time.perf_counter()

        for epoch in range(primeira_epoca, t["epochs"] + 1):
            model.train()
            epoch_loss, t_epoch = 0.0, time.perf_counter()

            componentes_epoca, n_amostras = {}, 0
            for lote in train_loader:
                alvo = self.loss.build_targets(lote, self.device)
                entrada = self.loss.build_inputs(lote, self.device)

                optimizer.zero_grad()
                loss, componentes = self.loss(model(entrada), alvo)
                loss.backward()
                if t.get("grad_clip"):
                    # o Eixo 2 da ablação pede gradient clipping ligado/desligado; sem a
                    # chave no config, nada é recortado
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                optimizer.step()

                n = len(lote)
                n_amostras += n
                epoch_loss += loss.item() * n
                for k, v in componentes.items():
                    componentes_epoca[k] = componentes_epoca.get(k, 0.0) + v * n

            train_loss = epoch_loss / n_amostras
            componentes_epoca = {k: v / n_amostras for k, v in componentes_epoca.items()}
            metricas = self._evaluate(model, val_loader)

            dt = time.perf_counter() - t_epoch
            history.append({
                "epoch": epoch, "train_loss": train_loss, "seconds": dt,
                **metricas,
                **{f"loss_{k}": v for k, v in componentes_epoca.items()},
            })

            # com várias perdas somadas, o total sozinho esconde qual delas estagnou
            detalhe = ""
            if len(componentes_epoca) > 1:
                detalhe = " (" + " ".join(f"{k} {v:.4f}" for k, v in componentes_epoca.items()) + ")"
            resumo = " ".join(f"{k} {v:.4f}" for k, v in metricas.items())
            print(f"época {epoch:3d}/{t['epochs']} | treino {train_loss:.4f}{detalhe} "
                  f"| {resumo} | {dt:.1f}s")

            if self.monitor not in metricas:
                raise KeyError(
                    f"train.monitor={self.monitor!r} não está entre as métricas de "
                    f"validação: {sorted(metricas)}"
                )
            if self._melhor(metricas[self.monitor], melhor):
                melhor = metricas[self.monitor]
                self._save_checkpoint(cfg.output_dir / "best.pth", model, optimizer,
                                      epoch, melhor, history)

            # `last.pth` é o ponto de retomada; `best.pth` é o modelo que vai para a
            # avaliação. São arquivos diferentes de propósito: o melhor modelo pode ser
            # de uma época bem anterior à última.
            if epoch % save_every == 0 or epoch == t["epochs"]:
                self._save_checkpoint(cfg.output_dir / "last.pth", model, optimizer,
                                      epoch, melhor, history)

        total = time.perf_counter() - t_start
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\ntempo total: {total/60:.1f} min ({total/t['epochs']:.1f}s por época)")
        print(f"melhor {self.monitor} de validação: {melhor:.4f}")
        print(f"checkpoint: {cfg.output_dir/'best.pth'}")
        print(f"retomável:  {cfg.output_dir/'last.pth'} (--resume)")

    def run_silencioso(self) -> None:
        """``run`` sem imprimir nada — para a ablação da Parte 3.

        São 36 runs de 40 épocas: o log por época daria 1.440 linhas e esconderia a única
        coisa que interessa acompanhar, que é o progresso da grade. O ``history.json`` e os
        checkpoints continuam sendo gravados igual, então nada de resultado se perde.
        """
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            self.run()

    @torch.no_grad()
    def _evaluate(self, model, loader) -> dict:
        """Métricas de validação da época. Sempre inclui ``val_loss``."""
        model.eval()
        total, n_amostras = 0.0, 0
        for lote in loader:
            alvo = self.loss.build_targets(lote, self.device)
            entrada = self.loss.build_inputs(lote, self.device)
            loss, _ = self.loss(model(entrada), alvo)
            n = len(lote)
            total += loss.item() * n
            n_amostras += n
        return {"val_loss": total / n_amostras}
