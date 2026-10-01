# Identidade ao longo do tempo: detecção, recorrência e rastreamento

Programming Assignment 2 — Aprendizado Profundo (FGV).

Produzir rótulos **identity-aware** em vídeo: se um objeto aparece no quadro 3 e reaparece
no quadro 40, ele tem que sair com o mesmo identificador. Sem rastreador pronto — o modelo
temporal, as perdas, a associação, a gestão de tracks e as métricas são de autoria nossa.

No PA1 fomos de *class-aware* para *instance-aware* no espaço; aqui vamos de
*instance-aware* por quadro para *identity-aware* no tempo.

## O resultado, em uma tabela

MOT17-10 (validação), detecções públicas SDP **congeladas**, mesma gestão de tracks nos três:

| rastreador | IDF1 | IDP | IDR | ID switches | MOTA |
|---|---|---|---|---|---|
| IoU, posição constante (Parte 1) | 0,4224 | 0,475 | 0,380 | 463 | 0,6479 |
| IoU + velocidade constante | 0,3839 | 0,434 | 0,344 | 487 | 0,6462 |
| **RNN como modelo de movimento (Parte 2)** | **0,4724** | **0,532** | **0,425** | **410** | **0,6530** |

+11,8% de IDF1 e −11,4% de ID switches sobre o baseline do enunciado, e +23,1% sobre o
baseline honesto de velocidade constante. A detecção é a mesma nos três (AP@0,5 = 0,8072):
**tudo que muda é o que acontece entre os quadros.**

## Setup

Duas famílias de comandos, conforme você tem `uv` instalado ou não. Elas fazem a mesma
coisa e instalam nos mesmos lugares (`.venv/`); use uma ou outra, sem misturar.

```bash
make setup        # com uv, CUDA
make setup-cpu    # com uv, sem pacote de placa de vídeo
make setup-pip    # sem uv: venv + pip + requirements.txt versionado
```

Este projeto **roda inteiro em CPU**. O modelo temporal tem ~19 mil parâmetros e o treino
leva 1 minuto; a ablação de 36 runs leva cerca de 50 minutos. Não há GPU no caminho crítico.

## Download dos dados

```bash
make dados          # anotações do MOT17 (~10 MB) — destrava tudo menos as figuras com quadro
make dados-imagens  # o pacote completo (5,86 GB) e as imagens das 7 sequências de treino
```

O primeiro comando é o que importa: **as Partes 0 a 5 rodam inteiras sobre texto.** O
`gt.txt` e o `det.txt` do MOT17 têm 10 MB juntos, e a trilha A opera sobre caixas, não sobre
imagens. Os 5,86 GB só são necessários para as figuras da Parte 4 e para o vídeo do
`notebooks/inferencia.ipynb`.

O `make dados-imagens` extrai só `MOT17-*-SDP/img1` (857 MB). As imagens são **idênticas**
nas três variantes de detector — só o `det.txt` muda —, então extrair as 21 pastas
triplicaria o disco sem acrescentar um pixel.

## Rodar

```bash
# Parte 0 — vídeos sintéticos, gt.txt em formato MOT, figura da oclusão
uv run python -m main --mode gen-synth --synthetic
# Parte 0 — piso fácil + varredura dos botões (o ensaio da Parte 1)
uv run python -m main --mode sweep --synthetic

# Parte 1 — baseline por quadro no MOT17
uv run python -m main --mode baseline --config configs/mot17.yaml --split trainval

# Parte 2 — UM COMANDO QUE TREINA
uv run python -m main --mode train --config configs/mot17_gru.yaml
# Parte 2 — UM COMANDO QUE AVALIA
uv run python -m main --mode eval  --config configs/mot17_gru.yaml

# Parte 3 — ablação do Eixo 1 (3 células × 4 janelas × 3 seeds)
uv run python -m main --mode ablation --config configs/mot17_gru.yaml
# Parte 4 — galeria de falhas e horizonte de memória
uv run python -m main --mode fails --config configs/mot17_gru.yaml --split trainval
# Parte 5 — queda de taxa de quadros, sem retreinar
uv run python -m main --mode stress --config configs/mot17_gru.yaml

uv run pytest tests/ -q
```

Sem `uv`, ative o `.venv` (`source .venv/bin/activate`) e tire o prefixo `uv run`.

## Os dados e o split

O `test/` do MOT17 não tem `gt.txt` (o gabarito ficou com os organizadores do benchmark),
então **o nosso split inteiro sai das 7 sequências de `train/`**. Medido em 30/09:

| seq | quadros | tamanho | ids | densidade | fps | câmera | movimento |
|---|---|---|---|---|---|---|---|
| MOT17-02 | 600 | 1920×1080 | 62 | 31,0 | 30 | parada | 0,0035 |
| MOT17-04 | 1050 | 1920×1080 | 83 | 45,3 | 30 | parada | 0,0043 |
| MOT17-05 | 837 | **640×480** | 133 | 8,3 | **14** | móvel | — |
| MOT17-09 | 525 | 1920×1080 | 26 | 10,1 | 30 | parada | — |
| MOT17-10 | 654 | 1920×1080 | 57 | 19,6 | 30 | móvel | 0,0290 |
| MOT17-11 | 900 | 1920×1080 | 75 | 10,5 | 30 | móvel | 0,0126 |
| MOT17-13 | 750 | 1920×1080 | 110 | 15,5 | 25 | móvel | 0,0597 |

*movimento* = deslocamento mediano do centro da caixa entre quadros, **em unidades da
altura da própria caixa** (ver `src/evaluation/dificuldade.py`).

**Split por sequência, nunca por quadro** — separar quadros ao acaso põe o quadro *t* no
treino e o *t+1* na validação, e o modelo temporal seria avaliado sobre algo que já viu:

| split | sequências | critério |
|---|---|---|
| treino | 02, 04, 11, 13 | as duas câmeras, densidade de 10 a 45 |
| validação | 10 | móvel, densidade no meio — escolha de época e de hiperparâmetro |
| **teste** | 09, 05 | 09 parada e esparsa; **05 fora da distribuição** em dois eixos |

A MOT17-05 é a única sequência com outra resolução (640×480) **e** outra taxa de quadros
(14 fps). Ela mede se o modelo aprendeu movimento ou decorou a escala do MOT17 — e conversa
direto com a Parte 5, que é sobre taxa de quadros.

### Duas armadilhas do formato MOT17, medidas e não supostas

**A sétima coluna do `gt.txt` não é confiança: é um flag** (1 = entra na avaliação,
0 = ignorar), e a oitava é a classe. Só `1, 1` (pedestre considerado) conta. Em MOT17-02
isso descarta **38% do arquivo** — 83 → 62 identidades, 30.003 → 18.581 caixas. O resto são
pessoas estáticas, occluders, bicicletas e distratores, que nenhum detector de pedestre vai
achar. Ver `read_mot_gt(only_active_pedestrians=True)`.

**O `det.txt` tem 7 campos, não 10**, e o `id` é sempre −1. Por isso existe
`src/data/detections.py` com um tipo próprio: detecção não tem identidade, e passá-la por
`Frame` levanta erro de id repetido no primeiro quadro com duas caixas — com razão.

## Qual detector público, e por quê

O MOT17 traz três (DPM, FRCNN, SDP). **Usamos o SDP.** AP@0,5 medida com o gabarito
filtrado:

| sequência | DPM | FRCNN | **SDP** |
|---|---|---|---|
| MOT17-02 | 0,3251 | 0,4115 | **0,5202** |
| MOT17-09 | 0,6983 | 0,6983 | **0,7955** |
| MOT17-10 | 0,4494 | 0,6192 | **0,8072** |

Ele ganha nas três e por margem larga. A escolha importa porque **o PA2 é sobre
associação**: com o detector mais forte, o que sobra de erro é de identidade, que é o
assunto. Com o DPM, metade do fracasso seria detecção, e a Parte 2 estaria consertando o
problema errado.

O DPM tem um agravante medido: os scores dele vão de **−0,50 a 3,14** (escala log-odds),
contra [0,40; 1,00] do SDP. Um limiar de confiança único significaria coisas diferentes em
cada um.

## Parte 0 — testes sintéticos

Quatro artefatos, três deles reaproveitados nas partes seguintes.

**1. Gerador.** Vídeos 128×128 de 30 a 60 quadros, 5 a 15 elipses com estado persistente
(id, posição, velocidade, cor, profundidade). A oclusão é **real, não aparente**: as elipses
são desenhadas em ordem de profundidade e a visibilidade é medida comparando a máscara que o
objeto teria sozinho com o que sobra dela. A duração é um **botão**, calibrado por um poste
de largura `occlusion_duration · speed + 2·meia_largura`.

**2. Simulador de detector** (`src/data/detector_sim.py`). Descarta `p%` das caixas, soma
ruído gaussiano com σ em fração da diagonal da caixa, injeta falsos positivos com tamanho
amostrado dos objetos reais. Com os três botões em zero, devolve o gabarito **bit a bit**.

**3. `metrics.py`.** IDF1, ID switches, fragmentações, erro de contagem e MOTA — nossos.
Ver a seção própria abaixo.

**4. O piso fácil e a varredura.** Três elipses, 1 px/quadro, sem oclusão:
**IDF1 = 0,9979, zero ID switches, zero erro de contagem.**

Depois, girando um botão por vez a partir do piso (`--mode sweep`):

| eixo | IDF1 | assinatura secundária | mecanismo da falha |
|---|---|---|---|
| objetos 3 → 20 | 0,998 → 0,998 | nada | **densidade sozinha não quebra nada** |
| velocidade 1 → 8 px/quadro | 0,998 → **0,719** | **102 ID switches** | a IoU entre quadros vizinhos some |
| oclusão 0 → 30 quadros | 0,998 → **0,573** | 0–5 switches, erro de contagem | a track morre e renasce com id novo |
| detector, 0 → 60% descartado | 0,998 → **0,527** | **108 fragmentações** | a track pisca |

**Quedas de IDF1 parecidas, três assinaturas completamente diferentes.** É exatamente por
isso que o enunciado exige ID switches e fragmentações contados explicitamente: o IDF1
sozinho não distingue um rastreador que troca rótulos de um que os perde.

### O oráculo não pode enxergar através das coisas

A caixa do gabarito é **amodal** — cobre o objeto inteiro mesmo quando ele está escondido.
Isso é propriedade da *anotação*, não do que existe na imagem. Servi-la como detecção cria um
detector que vê através das coisas, e com ele **girar o botão de oclusão não mede nada**:
medimos IDF1 = 1,0000 para oclusões de 0, 5, 10 e 20 quadros.

Por isso `detections_from_sequence` e `simulate_detections` recebem `min_visibility`. Um
objeto com `visibility = 0` não põe um pixel na tela; nenhum detector devolve caixa para ele.

## As métricas (`metrics.py`)

Implementação em `src/metrics/` — `iou.py`, `matching.py`, `identity.py`. `motmetrics`,
`TrackEval` e `py-motmetrics` são proibidos pelo enunciado e não são usados **nem como
referência nos testes**: conferir com a biblioteca proibida seria contornar a proibição pela
porta dos fundos.

**IDF1** é uma atribuição **global um-para-um** entre identidades previstas e verdadeiras,
decidida uma vez sobre a sequência inteira. Custo de casar *i* com *j* é
`|i| + |j| − 2·overlap(i,j)`, resolvido com Hungarian sobre uma matriz quadrada com
preenchimento em que **não casar também tem preço**. `IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)`.

**ID switches** usa um casamento por quadro que **preserva o par do quadro anterior** quando
ele ainda vale (família CLEAR-MOT). Sem isso, dois pedestres que se cruzam trocam de par por
geometria e a métrica conta switches que ninguém cometeu — a métrica estaria medindo a si
mesma.

**Fragmentações** são **retomadas**: a identidade era rastreada, deixou de ser, e voltou a
ser. Uma interrupção que nunca se recupera não conta, porque é indistinguível da track
acabando. Consequência a assumir: o número aqui é menor que o de implementações que contam
toda interrupção.

### Como sabemos que estão certas

```bash
uv run pytest tests/test_metrics.py -v
```

Os **três casos construídos à mão** que o enunciado exige, com a conta feita no próprio
teste:

| caso | IDF1 | ID switches | erro de contagem |
|---|---|---|---|
| (a) predição = gabarito | **1,0** exato | 0 | 0 |
| (b) duas identidades trocadas no quadro 5 | **0,5** | **2** | 0 |
| (c) uma track partida no quadro 3 de 10 | **0,7** | 1 | **1** |

Partir não é trocar, e as três medidas discordam — que é o ponto do caso (c).

Mais dois níveis de verificação:

- **verificador independente por força bruta**: enumera *todas* as atribuições possíveis e
  concorda com o Hungarian em 12 cenários sorteados e nos 3 casos acima. Algoritmo
  diferente, mesma definição — pega erro na montagem da matriz com preenchimento, que é o
  pedaço mais fácil de errar e o mais silencioso;
- **em escala real**: o gabarito do MOT17 alimentado como predição dá **IDF1 = 1,000 exato**
  nas 7 sequências (112 mil caixas), 0 switches, MOTA 1,000.

## Parte 1 — baseline por quadro

`IoUTracker`: IoU entre as detecções de *t* e a última caixa conhecida de cada track,
Hungarian, id novo quando nada casa, track morta depois de `max_age` quadros.
**NMS próprio** (`src/tracking/nms.py`) — `torchvision.ops.nms` é proibido.

**A regra de associação e a gestão de tracks** ficam em `TrackManager`
(`src/tracking/base.py`), e são **compartilhadas com o rastreador recorrente**. Se cada um
gerisse tracks do seu jeito, a comparação da Parte 2 mediria as duas coisas ao mesmo tempo:

| botão | valor | o que decide |
|---|---|---|
| `iou_threshold` | 0,3 | a 30 fps o pedestre anda pouco, mas a caixa do SDP oscila; 0,5 mata associação correta (IDF1 0,4553 contra 0,5024) |
| `max_age` | 10 | teto duro do horizonte de memória. Ajustado na **validação** |
| `min_hits` | 3 | filtra o falso positivo isolado, que viraria uma identidade inteira |
| `associacao` | hungarian | ótimo no quadro |

`min_hits` **emite retroativamente**: quando a track é confirmada, as caixas guardadas vão
para os quadros de onde vieram. Jogá-las fora custaria `min_hits − 1` quadros de toda
identidade em IDF1, de graça, já que a avaliação é offline.

### O gráfico do descolamento

`outputs/figures/p1_descolamento_trainval.png`. Treino + validação (o teste fica intocado):

| sequência | câmera | AP@0,5 | IDF1 | descolamento | ids previstas ÷ verdadeiras |
|---|---|---|---|---|---|
| MOT17-02 | parada | 0,5202 | 0,3743 | −0,146 | 3,0× |
| MOT17-04 | parada | 0,7510 | 0,7165 | −0,034 | 1,4× |
| MOT17-11 | móvel | 0,8261 | 0,6097 | −0,216 | 1,7× |
| MOT17-10 | móvel | 0,8072 | 0,4091 | −0,398 | 3,7× |
| MOT17-13 | móvel | 0,5548 | 0,4022 | −0,153 | 2,9× |

**957 identidades previstas para 387 verdadeiras.** A detecção resolve; a identidade não.

### O eixo de dificuldade: densidade não funciona

O enunciado deixa escolher entre densidade, movimento de câmera e duração de oclusão.
Medimos os três. **Densidade correlaciona ao contrário** com o descolamento (r = −0,61): a
MOT17-04 é a mais densa de todas (45 pedestres por quadro) e é a melhor rastreada, porque é
uma praça onde quase todo mundo está parado.

O eixo usado é o **deslocamento por quadro em unidades de altura da caixa**, que é livre de
escala. Ele **separa os dois grupos sem ambiguidade** (paradas em 0,0035 e 0,0043; móveis em
0,0126, 0,0290 e 0,0597 — fator de 3 a 14) mas **não ordena dentro de cada grupo**. Com 5
sequências não há como dizer se isso é falta de sinal ou falta de amostra, e está declarado
assim em vez de escolhido um eixo que ordenasse por acaso.

### Regras diferentes dão números diferentes

| variante | IDF1 | ID switches | erro de contagem |
|---|---|---|---|
| config (Hungarian, limiar 0,3, max_age 30) | 0,5024 | 1881 | 570 |
| guloso | 0,5018 | 2127 | 681 |
| limiar 0,5 | 0,4553 | 3450 | 1234 |
| max_age 5 | 0,5006 | 1445 | 770 |
| velocidade constante | 0,4636 | 1929 | 1159 |

Achado que vale a apresentação: **o guloso às vezes ganha do Hungarian.** No sintético
denso, IDF1 0,2217 contra 0,2156. Não é ruído — o Hungarian maximiza a soma das IoU *do
quadro*, que não é a quantidade que o IDF1 mede. Ser ótimo por quadro não é ser ótimo para a
identidade, e isso é o assunto do PA2.

## Parte 2 — trilha A: RNN como modelo de movimento

Um estado recorrente **por track**. A cada quadro ele recebe a última observação e prevê a
caixa do quadro seguinte; a associação usa IoU entre prevista e observada. Sob oclusão, o
estado roda para a frente sem observação.

### A decisão que faz funcionar: `forward` **é** o laço do rastreador

Não é parecido — é o mesmo código. A cada passo o modelo decide, pela máscara de observação,
se atualiza a crença com a caixa que chegou ou com a própria previsão anterior, que é
literalmente o que o `RNNTracker` faz quadro a quadro. Treinar com um laço e inferir com
outro é como se cria o descolamento de distribuição do Eixo 2 do enunciado.

### O que a rede prevê

O **incremento**, na parametrização do R-CNN:
`tx = (cx' − cx)/w`, `ty = (cy' − cy)/h`, `tw = log(w'/w)`, `th = log(h'/h)`.

**Incremento e não posição absoluta**: prever a caixa absoluta deixaria a rede memorizar
onde os pedestres costumam estar em cada sequência — pista real, que funciona na validação e
desaparece numa câmera nova.

**Dividido pelo tamanho da caixa e não do quadro**: um pedestre ao fundo anda 2 px por
quadro e o mesmo pedestre em primeiro plano anda 40. É o mesmo passo — o que mudou foi a
distância da câmera. Verificado: o mesmo movimento relativo em escalas 10× diferentes produz
alvo idêntico.

### As 11 entradas por passo

`tx ty tw th` (4, o movimento observado) · `cx cy w h` (4, dividido pelo tamanho do quadro) ·
`dt` (1) · `observado` (1) · `score` (1).

**`dt` em segundos, não em quadros.** O MOT17 tem sequências a 14, 25 e 30 fps; em quadros,
"um passo" significaria três coisas diferentes no mesmo dataset.

### A perda

`smooth-L1` sobre o **resíduo** `delta(caixa_verdadeira, caixa_prevista)`, que é zero quando
as duas coincidem. O denominador é o tamanho da caixa verdadeira, que não depende do modelo,
então o gradiente entra só pela previsão — e como a previsão de um passo vira a crença do
seguinte, ele atravessa o rollout inteiro. É o BPTT cuja norma a Parte 4 mede.

`peso_sob_oclusao: 3.0` concentra o gradiente nos passos às cegas. A perda de treino
confirma que são regimes diferentes: **0,0024 nos passos observados contra 0,0099 nos
passos cegos**, um fator de 4.

### Oclusão simulada no treino

Metade das janelas recebe um buraco **contíguo** (não passos soltos: oclusão real é
contígua). Sem isso o modelo nunca vê no treino o regime em que opera na inferência.

### A política de emissão, e o que ela custa

| | IoU | RNN | MOTA |
|---|---|---|---|
| `emitir_sem_observacao: false` | 0,4224 | **0,4724** | 0,65 |
| `emitir_sem_observacao: true` | 0,3109 | 0,3296 | **0,07** |

Sustentar a caixa durante o buraco melhora revocação (IDR 0,369 → 0,390) e corta
fragmentações em 25%, mas cada quadro emitido às cegas é um falso positivo se o objeto não
estava lá — e com `max_age` quadros de buraco isso enterra a precisão. **Vale para os dois
rastreadores**, então não é defeito da recorrência: é a política de decodificação.

### A pergunta da apresentação (vídeo de 2 h em janelas de T quadros)

O que quebra na fronteira entre janelas é o **estado oculto**, que é zerado — o análogo
temporal do objeto partido na divisa entre tiles do PA1. A trilha A permite costurar de duas
formas: carregar o `h` do fim de uma janela para o início da seguinte (o estado é o resumo, e
tem 55 números), e comparar a caixa prevista do fim da janela anterior com as detecções do
começo da seguinte. As duas são baratas justamente porque o que persiste é pequeno.

## Parte 3 — Eixo 1: a célula recorrente

RNN simples × LSTM × GRU, **mesmo orçamento de parâmetros**, com `T ∈ {4, 8, 16, 32}` e
**3 seeds**. O orçamento é igualado por busca e conferido, não calculado à mão: com teto de
20k, `rnn hidden 95` (19.764), `lstm hidden 48` (19.588), `gru hidden 55` (19.364) — entre
96,8% e 98,8% do teto.

Resultados em `outputs/ablation/run_*/report.md` e `eixo1.png`.

Este eixo foi escolhido porque a Parte 4 **reaproveita os checkpoints**: ela manda comparar a
curva de gradiente da RNN simples com a do modelo com portas na mesma janela.

## Parte 4 — galeria de falhas e horizonte de memória

### As duas medidas concordam

| medida | valor |
|---|---|
| **analítica** — até onde `‖∂L_t/∂h_{t−k}‖` chega | **6 passos** (o gradiente cai 67× até lá) |
| **empírica** — até quantos quadros de buraco a identidade volta | **5 quadros** |
| **a régua** — oclusão do dataset | média 20,4 quadros, p90 46 |

**63% das oclusões do dataset são mais longas que o horizonte.** A curva analítica é uma reta
em escala log — decaimento exponencial, a história de gradiente que some contada na aula,
medida no nosso modelo.

### O estimador empírico que estava errado

A primeira versão pegava a maior duração com taxa de sobrevivência ≥ 50%, sem mínimo de
amostra, e devolveu **36 quadros** num modelo cujo gradiente morre em 6 — porque havia uma
oclusão de 36 quadros, uma só, que por acaso sobreviveu. Um evento não é uma taxa. Corrigido
com mínimo de 5 eventos por faixa e parada na primeira faixa que falha.

### A correção obrigatória

Diagnóstico da Parte 5: o modelo recebe Δt como entrada mas as sequências de treino são de 25
e 30 fps, então ele só viu **Δt entre 0,0333 e 0,0400 s — uma faixa de 1,2×**. Não há sinal
nisso para aprender a dependência. Mudança implementada: `strides_dt: [1, 2, 3, 5]` em
`configs/mot17_gru_dtaug.yaml`, que gera janelas subamostradas das mesmas trajetórias e faz o
Δt variar 5× dentro do treino (10.441 → 20.434 janelas). O antes/depois está abaixo.

## Parte 5 — queda de taxa de quadros

**Sem retreinar**, em cima do modelo final. Subamostrar é pular linhas das listas já em
memória: nenhum detector roda de novo.

| condição | 1/1 | 1/2 | 1/5 |
|---|---|---|---|
| IoU (baseline) | 0,4224 | 0,4105 | 0,2527 |
| RNN alheio ao Δt | 0,4724 | 0,4760 | 0,3298 |
| RNN informado do Δt | 0,4724 | 0,4740 | 0,3287 |

**Informar o Δt correto não recupera nada** (−0,7% da queda). A pergunta do enunciado
("alimentar Δt na recorrência resolveria?") tem resposta medida, e é **não** — com o
diagnóstico junto: a 1/5, o Δt pedido é 0,1667 s, **4,2× o maior valor que o modelo já viu no
treino**. Ter a entrada não é ter aprendido a usá-la.

Detalhe secundário: a 1/2 o RNN melhora de leve (0,4760 contra 0,4724). Subamostrar remove
alguns quadros de detecção ruim.

## Estrutura

```
metrics.py                  entregável nomeado no enunciado (re-export de src/metrics/)
main.py                     entry point único (--mode)
src/
  boxes.py                  parametrização da caixa (incremento estilo R-CNN)
  core/                     AppConfig (singleton) carregado do YAML
  data/     sequence.py · mot_format.py · detections.py · detector_sim.py
            synthetic_video.py · mot17.py · windows.py (janelas de BPTT)
            factory.py (registry) · pipeline.py (3 splits)
  metrics/  iou.py · matching.py (guloso/Hungarian/CLEAR-MOT) · identity.py · detection.py
  models/   motion_rnn.py (MotionRNN + orçamento de parâmetros) · checkpoint.py · factory.py
  losses/   motion.py (smooth-L1 sobre o resíduo) · factory.py
  tracking/ base.py (Track, TrackManager) · iou_tracker.py · rnn_tracker.py · nms.py
  training/ engine.py (loop, seed, checkpoint, resume)
  evaluation/ engine.py · tracking_engine.py · occlusion.py · dificuldade.py
  analysis/ memory.py (horizonte analítico e empírico)
  ablation/ runner.py (Eixo 1)
  fails/    runner.py (Parte 4)
  stress/   framerate.py (Parte 5)
  synthetic/ runner.py (gen-synth) · sweep.py (varredura dos botões)
```

## Licença

Ver `LICENSE`.
