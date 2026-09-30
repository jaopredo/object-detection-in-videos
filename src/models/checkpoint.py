"""Carregamento de pesos compatível com checkpoints de versões anteriores do código.

A refatoração que separou os modelos em ``Segmenter(backbone, cabeça)`` mudou os nomes dos
pesos no ``state_dict``, sem mudar nenhum tensor:

    antes                       agora
    encoders.0.block.0.weight   backbone.encoders.0.block.0.weight
    head.weight                 head.head.weight            (cabeça binária)
    seg_head.weight             head.seg_head.weight        (cabeça da Trilha C)

Os checkpoints salvos antes disso (Partes 0, 1 e 2) deixaram de carregar. Este módulo
reconhece o formato antigo e acrescenta o prefixo que falta. É só renomeação: a arquitetura
e os valores são os mesmos, e o carregamento continua estrito — qualquer chave que sobre, que
falte ou que tenha outro formato levanta erro, em vez de carregar pela metade em silêncio.
"""

from pathlib import Path

import torch
import torch.nn as nn

# Prefixos tentados, em ordem, para uma chave antiga que não existe no modelo atual.
_PREFIXOS = ("backbone.", "head.")


def remap_legacy_keys(state_dict: dict, model: nn.Module) -> tuple[dict, bool]:
    """Traduz um ``state_dict`` do formato antigo para o do modelo atual.

    Args:
        state_dict: pesos lidos do checkpoint.
        model: modelo que vai receber os pesos.

    Returns:
        Tupla (state_dict pronto para ``load_state_dict``, se houve remapeamento).

    Raises:
        KeyError: se alguma chave do checkpoint não corresponder a nenhuma do modelo, ou se
            alguma chave do modelo ficar sem valor. O checkpoint seria de outra arquitetura.
    """
    alvo = model.state_dict()
    if set(state_dict) == set(alvo):
        return state_dict, False

    traduzido = {}
    for chave, valor in state_dict.items():
        if chave in alvo:
            traduzido[chave] = valor
            continue
        for prefixo in _PREFIXOS:
            if prefixo + chave in alvo:
                traduzido[prefixo + chave] = valor
                break
        else:
            raise KeyError(
                f"a chave {chave!r} do checkpoint não corresponde a nenhuma do modelo "
                f"{type(model).__name__} — o checkpoint é de outra arquitetura ou outra cabeça"
            )

    faltando = sorted(set(alvo) - set(traduzido))
    if faltando:
        raise KeyError(
            f"o checkpoint não tem {len(faltando)} pesos que o modelo exige "
            f"(ex.: {faltando[0]!r}) — o checkpoint é de outra arquitetura ou outra cabeça"
        )
    return traduzido, True


def load_model_weights(model: nn.Module, state_dict: dict) -> bool:
    """Carrega os pesos no modelo, aceitando o formato antigo de nomes.

    Returns:
        ``True`` se foi preciso remapear nomes antigos.
    """
    traduzido, remapeado = remap_legacy_keys(state_dict, model)
    model.load_state_dict(traduzido)   # estrito: formas incompatíveis também levantam erro
    return remapeado


def load_checkpoint(path, model: nn.Module, device="cpu") -> dict:
    """Lê um checkpoint do disco e carrega os pesos no modelo.

    Returns:
        O dicionário completo do checkpoint (época, histórico, config, otimizador...).
    """
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    if load_model_weights(model, ck["model"]):
        print(f"checkpoint {path}: nomes de pesos no formato antigo, remapeados")
    return ck
