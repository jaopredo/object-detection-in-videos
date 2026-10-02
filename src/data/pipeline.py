"""Monta os três conjuntos a partir da configuração."""

from typing import NamedTuple

from src.core.config import AppConfig
from src.data.factory import DatasetFactoryRegistry


class Splits(NamedTuple):
    """Os três conjuntos, nomeados.

    Uma tupla de três posições seria fácil de trocar de ordem sem ninguém perceber — e
    trocar validação com teste é o tipo de erro que só aparece quando o número da
    apresentação está errado. Com nomes, ``splits.test`` é uma escolha explícita.
    """

    train: object
    val: object
    test: object


class DataPipeline:
    """Constrói os conjuntos; não empacota em DataLoader.

    No PA1 esta classe devolvia DataLoaders, porque a amostra era uma imagem e o lote era
    óbvio. Aqui a amostra é uma **sequência** de comprimento variável, e as sequências não
    são o que se passa para uma rede: a Parte 2 vai treinar sobre janelas recortadas delas.
    Empacotar sequências num DataLoader agora só criaria um collate artificial para
    desmontar depois.
    """

    def __init__(self, app_config: AppConfig, config=None):
        self.cfg = config or app_config.get_train_config()
        self.factory = DatasetFactoryRegistry.get(app_config)

    def build_datasets(self) -> Splits:
        return Splits(
            train=self.factory.build_train(self.cfg),
            val=self.factory.build_val(self.cfg),
            test=self.factory.build_test(self.cfg),
        )

    def build_split(self, split: str):
        """Um conjunto só — para avaliar sem pagar a geração dos outros dois."""
        return self.factory.build(self.cfg, split)

    def build_dataloaders(self, train_split: str = "train", val_split: str = "val"):
        """Lotes de janelas de trajetória para treinar o modelo temporal (Parte 2).

        A amostra é uma janela de ``T`` quadros de **uma** identidade — ver
        ``src/data/windows.py``. O ``T`` vem de ``train.window``, e é o botão do Eixo 1 da
        ablação da Parte 3.

        A validação não recebe oclusão simulada (``p_oclusao = 0``): a perda de validação
        escolhe a época, e se o buraco fosse sorteado a cada avaliação a métrica mudaria de
        uma época para outra por motivo que não é o modelo. O regime sob oclusão é medido de
        propósito na Parte 4, com o rastreador inteiro e IDF1.

        ``train_split``/``val_split`` default para o split oficial do projeto. O grid search
        de hiperparâmetros passa ``"grid_train"``/``"grid_val"`` para treinar e validar sobre
        o holdout interno (MOT17-02), sem tocar no ``val`` oficial (MOT17-10).
        """
        from torch.utils.data import DataLoader

        from src.data.windows import TrackWindows, collate

        t = self.cfg.train
        train_dataset = self.build_split(train_split)
        val_dataset = self.build_split(val_split)
        janela = t.get("window", 16)

        def monta(dataset, p_oclusao, embaralha):
            sequencias = [dataset[i] for i in range(len(dataset))]
            janelas = TrackWindows(
                sequencias, T=janela, stride=t.get("stride"),
                p_oclusao=p_oclusao, oclusao_max=t.get("oclusao_max", 8),
                strides_dt=tuple(t.get("strides_dt", (1,))),
            )
            return DataLoader(
                janelas, batch_size=t.get("batch_size", 128), shuffle=embaralha,
                collate_fn=collate, num_workers=t.get("num_workers", 0),
            )

        return (
            monta(train_dataset, t.get("p_oclusao", 0.5), True),
            monta(val_dataset, 0.0, False),
        )
