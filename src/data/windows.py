"""Janelas de trajetória — o que a Parte 2 treina, e o que ``build_dataloaders`` faltava.

A amostra do PA2 **não é um quadro**. É um pedaço de trajetória de um objeto só: T caixas
consecutivas da mesma identidade, que é o comprimento sobre o qual o BPTT vai propagar
gradiente. ``T`` é o botão do Eixo 1 da Parte 3.

Duas decisões de recorte estão aqui:

**Uma janela pertence a uma identidade.** Não se mistura objeto numa janela. O estado
recorrente do rastreador é por track, então treinar sobre qualquer outra coisa seria treinar
um modelo que não é o que se usa.

**A oclusão é simulada no treino, e é contígua.** Uma fração das janelas recebe um buraco:
alguns passos seguidos marcados como não observados, durante os quais o modelo se alimenta
da própria previsão. Sem isso ele nunca vê, no treino, o regime em que opera sob oclusão —
que é o descolamento de distribuição que o Eixo 2 do enunciado descreve. O buraco é
**contíguo**, e não passos soltos ao acaso, porque oclusão real é contígua: o pedestre passa
atrás do poste e fica lá.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.boxes import xyxy_para_cxcywh
from src.data.sequence import Sequence


@dataclass
class JanelaLote:
    """Um lote de janelas, já em tensores.

    É uma classe e não um dicionário por causa de ``__len__``: o ``TrainEngine`` usa
    ``len(lote)`` para ponderar a perda pelo tamanho do lote, e ``len`` de um dicionário é o
    número de chaves — daria 5 sempre, e a média das épocas sairia errada em silêncio.
    """

    caixas: torch.Tensor      #: (B, T, 4) cxcywh em pixels
    observado: torch.Tensor   #: (B, T) 1 se o passo é observado
    dt: torch.Tensor          #: (B, T) segundos desde o passo anterior
    score: torch.Tensor       #: (B, T)
    tamanho: torch.Tensor     #: (B, 2) largura e altura do quadro

    def __len__(self) -> int:
        return len(self.caixas)

    def to(self, device) -> "JanelaLote":
        return JanelaLote(*(t.to(device) for t in
                            (self.caixas, self.observado, self.dt, self.score, self.tamanho)))


def trechos_continuos(sequence: Sequence, track_id: int) -> list[list[int]]:
    """Quadros em que a identidade aparece, quebrados em trechos **sem buraco**.

    Buraco no gabarito é ausência de caixa, e sem caixa não há alvo. Treinar por cima de um
    buraco exigiria inventar o alvo; recortar em trechos contínuos e simular a oclusão
    depois é honesto e dá controle sobre a duração do buraco — que é o que a Parte 4 mede.
    """
    presentes = [t for t, f in enumerate(sequence) if track_id in set(f.ids.tolist())]
    if not presentes:
        return []

    trechos, atual = [], [presentes[0]]
    for anterior, agora in zip(presentes[:-1], presentes[1:]):
        if agora == anterior + 1:
            atual.append(agora)
        else:
            trechos.append(atual)
            atual = [agora]
    trechos.append(atual)
    return trechos


class TrackWindows(Dataset):
    """Janelas de ``T`` quadros recortadas das trajetórias do gabarito.

    Args:
        sequences: as sequências do split.
        T: comprimento da janela — a janela de BPTT truncado do Eixo 1.
        stride: passo entre janelas. Menor que ``T`` faz as janelas se sobreporem, o que
            multiplica as amostras sem inventar dado; ``None`` usa ``max(1, T // 2)``.
        p_oclusao: fração das janelas que recebe um buraco simulado.
        oclusao_max: duração máxima do buraco, em passos. Limitado a ``T - 2``: uma janela
            precisa do primeiro passo observado (para o estado começar de algum lugar) e de
            pelo menos um alvo depois do buraco.
        strides_dt: fatores de subamostragem usados para gerar janelas **adicionais**, em
            que se pega um quadro a cada ``k`` e o Δt vale ``k/fps``.

            É a correção que o diagnóstico da Parte 5 pede, e a razão merece o espaço.
            O modelo recebe Δt como entrada desde o começo — mas as sequências de treino
            são de 25 e 30 fps, então ele só viu Δt entre 0,0333 e 0,0400 s: **uma faixa de
            1,2x**. Não há sinal nenhum nisso para aprender a dependência. Quando a Parte 5
            subamostra o vídeo a 1/5, ela pede Δt = 0,1667 s, que é 4,2x o maior valor já
            visto, e medimos que informar o Δt correto não recupera nada da queda (−0,7%).

            Ter a entrada não é ter aprendido a usá-la. Com ``strides_dt=(1, 2, 3, 5)`` as
            mesmas trajetórias produzem janelas em quatro taxas, e o Δt passa a variar 5x
            dentro do treino.
        seed: semente da simulação de oclusão. ``None`` usa o gerador global do torch, que é
            o que o ``TrainEngine`` fixa — então o treino inteiro continua reprodutível.
    """

    def __init__(
        self,
        sequences: list[Sequence],
        T: int = 16,
        stride: int | None = None,
        p_oclusao: float = 0.5,
        oclusao_max: int = 8,
        strides_dt: tuple[int, ...] = (1,),
        seed: int | None = None,
    ):
        self.T = T
        self.stride = stride if stride is not None else max(1, T // 2)
        self.p_oclusao = p_oclusao
        self.oclusao_max = min(oclusao_max, max(1, T - 2))
        self.strides_dt = tuple(strides_dt)
        self.gerador = torch.Generator().manual_seed(seed) if seed is not None else None

        self.janelas: list[dict] = []
        for seq in sequences:
            for (track_id, n_trecho), (quadros, caixas) in self._indexar(seq).items():
                for k in self.strides_dt:
                    # com subamostragem de k, a janela de T passos cobre k*T quadros do
                    # vídeo original, então o trecho precisa ser k vezes mais longo
                    alcance = (T - 1) * k + 1
                    for inicio in range(0, len(quadros) - alcance + 1, self.stride * k):
                        self.janelas.append({
                            "caixas": caixas[inicio:inicio + alcance:k],
                            # o dt da janela é o que o modelo vai receber: k quadros de
                            # intervalo valem k/fps segundos
                            "dt": k / seq.fps,
                            "tamanho": np.array([seq.width, seq.height], dtype=np.float32),
                            "sequencia": seq.name,
                            "track_id": track_id,
                            "trecho": n_trecho,
                            "stride_dt": k,
                            "quadro_inicial": int(quadros[inicio]),
                        })

    @staticmethod
    def _indexar(seq: Sequence) -> dict[tuple[int, int], tuple[list[int], np.ndarray]]:
        """``(identidade, nº do trecho)`` → (quadros, caixas em ``cxcywh``).

        A chave inclui o número do trecho porque dois trechos da mesma identidade são
        trajetórias separadas por um buraco do gabarito. Indexar só pela identidade faria o
        segundo trecho sobrescrever o primeiro — e a metade dos dados sumiria em silêncio.
        """
        saida = {}
        for track_id in seq.track_ids:
            for n, trecho in enumerate(trechos_continuos(seq, int(track_id))):
                caixas = np.stack([seq[t].box_of(int(track_id)) for t in trecho])
                # a chave inclui o número do trecho: dois trechos da mesma identidade são
                # trajetórias separadas, e juntá-las criaria uma janela que atravessa um
                # buraco do gabarito
                saida[(int(track_id), n)] = (trecho, xyxy_para_cxcywh(caixas))
        return saida

    def __len__(self) -> int:
        return len(self.janelas)

    def __getitem__(self, i: int) -> dict:
        janela = self.janelas[i]
        T = self.T

        observado = torch.ones(T)
        if self.p_oclusao > 0 and self._sorteio() < self.p_oclusao:
            duracao = 1 + int(self._sorteio() * self.oclusao_max)
            # começa em 1 no mínimo: o passo 0 tem que ser observado, senão o estado não
            # tem de onde partir e a janela vira ruído
            inicio = 1 + int(self._sorteio() * max(1, T - duracao - 1))
            observado[inicio:min(inicio + duracao, T)] = 0.0

        return {
            "caixas": torch.from_numpy(janela["caixas"]).float(),
            "observado": observado,
            # dt em SEGUNDOS: o MOT17 tem sequências a 14, 25 e 30 fps, e em quadros "um
            # passo" significaria três coisas diferentes no mesmo dataset
            "dt": torch.full((T,), janela["dt"]),
            "score": observado.clone(),   # gabarito: confiança 1 onde se vê, 0 onde não
            "tamanho": torch.from_numpy(janela["tamanho"]),
        }

    def _sorteio(self) -> float:
        return float(torch.rand(1, generator=self.gerador).item())


def collate(amostras: list[dict]) -> JanelaLote:
    """Empilha as amostras num ``JanelaLote``."""
    junta = lambda chave: torch.stack([a[chave] for a in amostras])
    return JanelaLote(
        caixas=junta("caixas"), observado=junta("observado"), dt=junta("dt"),
        score=junta("score"), tamanho=junta("tamanho"),
    )
