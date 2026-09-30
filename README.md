# Identidade ao longo do tempo: detecção, recorrência e rastreamento

Programming Assignment 2 — Aprendizado Profundo (FGV).

Produzir rótulos **identity-aware** em vídeo: se um objeto aparece no quadro 3 e reaparece
no quadro 40, ele tem que sair com o mesmo identificador. Sem rastreador pronto — o modelo
temporal, as perdas, a associação e a gestão de tracks são de autoria nossa.

No PA1 fomos de *class-aware* para *instance-aware* no espaço; aqui vamos de
*instance-aware* por quadro para *identity-aware* no tempo.

## Estado

| Parte | O que é | Estado |
|---|---|---|
| 0 | Testes sintéticos | gerador de vídeos ✅ · simulador de detector, IDF1 e baseline ⏳ |
| 1 | Baseline por quadro | ⏳ |
| 2 | Memória temporal (RNN) | ⏳ |
| 3 | Ablação | ⏳ |
| 4 | Galeria de falhas e horizonte de memória | ⏳ |
| 5 | Teste de estresse | ⏳ |

## Setup

Duas famílias de comandos, conforme você tem `uv` instalado ou não. Elas fazem a mesma
coisa e instalam nos mesmos lugares (`.venv/`); use uma ou outra, sem misturar.

**Com uv:**

```bash
make setup        # com CUDA
make setup-cpu    # sem pacote de placa de vídeo
```

**Sem uv** (usa `venv` + `pip` e o `requirements.txt` versionado):

```bash
make setup-pip
make setup-cpu-pip
```

A Parte 0 **não precisa de torch** — o gerador é numpy e scikit-image puros. O torch só
entra na Parte 2, com o modelo temporal.

## Rodar

```bash
# Parte 0 — gera os vídeos sintéticos, o gt.txt e a figura da oclusão (~1 min, CPU)
uv run python -m main --mode gen-synth --synthetic

# caracteriza um split: densidade, identidades, distribuição de duração de oclusão
uv run python -m main --mode eval --synthetic
uv run python -m main --mode eval --synthetic --split test   # só no fim

# testes
uv run pytest tests/ -v
```

Sem `uv`, ative o `.venv` (`source .venv/bin/activate`) e tire o prefixo `uv run`.

## O que sai da Parte 0

```
outputs/p0/sequences/SYNTH-TRAIN-00/
    img1/000001.png ...      quadros, numerados a partir de 1
    gt/gt.txt                anotações no formato MOTChallenge
    video.mp4                o mesmo conteúdo, para olhar
    seqinfo.ini              fps, tamanho, nº de quadros
outputs/figures/oclusao.png  a figura: uma identidade some por N quadros e volta
```

A saída imita a organização do MOTChallenge de propósito: o leitor do MOT17 da Parte 1 abre
estas pastas sem caminho especial, e qualquer script escrito contra o MOT17 funciona no
sintético de graça. É isso que permite fechar e depurar a Parte 1 inteira antes de baixar
os 5,5 GB do dataset real.

## Configuração

Um YAML por experimento, em `configs/`. `configs/synthetic.yaml` é o da Parte 0; os botões
do gerador são os que a Parte 1 vai girar para mostrar onde o baseline quebra:

```yaml
data:
  n_train: 8                # split POR SEQUÊNCIA — nunca por quadro
  n_val: 2
  n_test: 2
  min_obj: 5
  max_obj: 15
  speed: 2.0                # pixels por quadro
  occlusion_duration: 10    # quadros atrás do poste
```

Três conjuntos, não dois: a escolha de época e de hiperparâmetro acontece na **validação**,
e o **teste** só é tocado no fim — `--mode eval --split test` avisa em destaque quando isso
acontece.

## Estrutura

```
src/
  core/        AppConfig (singleton) carregado do YAML
  data/        sequence.py (Frame/Sequence) · mot_format.py · synthetic_video.py
               factory.py (registry + validação de chaves) · pipeline.py (3 splits)
  evaluation/  engine.py · occlusion.py (trechos e duração de oclusão)
  synthetic/   runner.py — o que --mode gen-synth executa
  training/    engine.py (loop, seed, checkpoint, resume) — usado a partir da Parte 2
  models/ losses/   registries vazios até a Parte 2
main.py        entry point único (--mode)
```

## Licença

Ver `LICENSE`.
