"""Onde e por quanto tempo cada identidade fica escondida.

A Parte 4 pede explicitamente para comparar o horizonte de memória do modelo com a
*distribuição de duração de oclusão do dataset*. Para isso a duração precisa ser um número
extraído dos dados, não uma impressão — e é o que este módulo produz, tanto do sintético
quanto do MOT17, já que ambos chegam aqui como ``Sequence``.

Uma oclusão é um trecho **contínuo** de quadros em que a identidade existe antes e depois,
mas não aparece no meio. O "antes e depois" importa: um objeto que ainda não nasceu, ou que
já saiu de cena para sempre, também tem visibilidade zero, e contar isso como oclusão
inflaria a distribuição com o que na verdade é nascimento e morte de track.
"""

from dataclasses import dataclass

import numpy as np

from src.data.sequence import Sequence


@dataclass
class OcclusionRun:
    """Um trecho contínuo em que uma identidade some e depois volta."""

    track_id: int
    start: int          #: primeiro quadro escondido
    end: int            #: último quadro escondido (inclusive)
    last_seen: int      #: último quadro em que apareceu antes de sumir
    next_seen: int      #: primeiro quadro em que reapareceu

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


def occlusion_runs(sequence: Sequence, threshold: float = 0.0) -> list[OcclusionRun]:
    """Todos os trechos de oclusão da sequência, de todas as identidades.

    Args:
        sequence: a sequência a analisar.
        threshold: visibilidade **no máximo** igual a isto conta como escondido. O padrão
            ``0.0`` é oclusão total; subir para ``0.1``, por exemplo, passa a contar
            também o objeto que só mostra uma nesga.

    Returns:
        Lista de ``OcclusionRun``, ordenada por identidade e depois por quadro inicial.
    """
    runs: list[OcclusionRun] = []

    for track_id in sequence.track_ids:
        visivel = sequence.visibility_of(track_id) > threshold
        aparicoes = np.flatnonzero(visivel)
        if len(aparicoes) < 2:
            # nunca apareceu, ou apareceu uma vez só: não há "some e volta" possível
            continue

        # o recorte entre a primeira e a última aparição é o que descarta nascimento e
        # morte — antes da primeira e depois da última não é oclusão, é ausência
        primeiro, ultimo = aparicoes[0], aparicoes[-1]

        inicio = None
        for t in range(primeiro + 1, ultimo + 1):
            if not visivel[t] and inicio is None:
                inicio = t
            elif visivel[t] and inicio is not None:
                runs.append(OcclusionRun(
                    track_id=int(track_id), start=inicio, end=t - 1,
                    last_seen=inicio - 1, next_seen=t,
                ))
                inicio = None

    return runs


def longest_run(sequence: Sequence, threshold: float = 0.0) -> OcclusionRun | None:
    """O trecho de oclusão mais longo da sequência, ou ``None`` se não houver nenhum."""
    runs = occlusion_runs(sequence, threshold)
    return max(runs, key=lambda r: r.duration) if runs else None


def sequence_stats(sequence: Sequence) -> dict:
    """Resumo numérico de uma sequência — o que vai para ``per_sequence.json``.

    As três primeiras chaves são os eixos de dificuldade que o enunciado sugere para
    ordenar as sequências nos gráficos da Parte 1 (densidade, movimento, oclusão).
    """
    runs = occlusion_runs(sequence)
    duracoes = [r.duration for r in runs]
    por_quadro = [len(f) for f in sequence]
    visibilidades = np.concatenate([f.visibility for f in sequence if len(f)]) \
        if any(len(f) for f in sequence) else np.zeros(1)

    return {
        "name": sequence.name,
        "n_frames": len(sequence),
        "n_identities": int(len(sequence.track_ids)),
        "density_mean": float(np.mean(por_quadro)) if por_quadro else 0.0,
        "density_max": int(np.max(por_quadro)) if por_quadro else 0,
        "visibility_mean": float(visibilidades.mean()),
        "n_occlusions": len(runs),
        "occlusion_mean": float(np.mean(duracoes)) if duracoes else 0.0,
        "occlusion_max": int(np.max(duracoes)) if duracoes else 0,
        "occlusion_durations": [int(d) for d in duracoes],
    }
