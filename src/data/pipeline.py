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

    def build_dataloaders(self):
        """Lotes de janelas de trajetória para treinar o modelo temporal.

        Ainda não existe: o recorte de janelas de BPTT depende de qual trilha da Parte 2
        for escolhida (movimento → sequência de caixas; aparência → sequência de
        embeddings), e inventar o formato antes dessa decisão seria adivinhar.
        """
        raise NotImplementedError(
            "o Dataset de janelas de BPTT chega na Parte 2 — ver src/data/windows.py. "
            "As partes 0 e 1 rodam sem treino: use --mode gen-synth ou --mode eval."
        )
