# Uso de IA neste assignment

Usamos o Claude Code (Anthropic) como assistente de implementação, revisão e explicação ao
longo de boa parte do PA2 — principalmente na Parte 2b (grid search, que não era pedida pelo
enunciado e foi ideia nossa), em correções de infraestrutura (organização de `outputs/`,
geração de vídeo), e numa revisão posterior que achou três problemas reais de metodologia
antes da apresentação.

## Episódios

### 1. Grid search de hiperparâmetros (Parte 2b)

O projeto não tinha nenhum mecanismo pra escolher `lr`/`batch_size` por validação — eram
fixados à mão no YAML. Pedimos pra IA desenhar e implementar uma varredura, e discutimos
junto várias decisões antes de qualquer linha de código:

- **Split de validação da busca**: não dava pra tirar "10% dos dados" de forma limpa com só
  4 sequências de treino de tamanhos desiguais — fechamos em reservar uma sequência inteira
  (MOT17-02, a menor) como holdout interno, separado do `val` oficial (MOT17-10).
- **Métrica de seleção**: IDF1 via tracking completo, não a perda de regressão por época.
- **Extensibilidade**: a IA sugeriu (e implementamos) overrides por dotted-path no YAML
  (`train.lr`, `model.param_budget`, ...), pra adicionar um hiperparâmetro na grade bastar
  editar o config, sem mexer em código.
- 3 seeds por combinação, mesma régua de confiabilidade que já usamos na ablação da Parte 3.

Resultado: `src/gridsearch/runner.py` novo, dois splits novos em `src/data/mot17.py`
(`grid_train`/`grid_val`), e um retreino final que grava no mesmo lugar que `make treinar`
(`outputs/p2`), pra `--mode eval`/`fails`/`stress` reaproveitarem o vencedor sem precisar de
flag nenhuma.

### 2. `outputs/p2/p4` e `outputs/p2/p5` — bug de organização

Percebemos que os resultados da Parte 4 e da Parte 5 estavam sendo gravados **dentro** da
pasta de saída da Parte 2 (`outputs/p2/p4/`, `outputs/p2/p5/`), em vez de pastas irmãs como
`outputs/p0` e `outputs/p1`. Pedimos pra IA investigar — a causa era as Partes 4/5
reaproveitarem o config (e o `output_dir`) da Parte 2 só pra achar o checkpoint, mas sem
separar "onde o modelo mora" de "onde o resultado desta parte vai". Corrigido em
`src/fails/runner.py` e `src/stress/framerate.py`: agora gravam na pasta **irmã** do
`output_dir`, não dentro dele.

### 3. Vídeo da galeria de falhas e do teste de estresse (extra, não pedido)

Depois de entender a Parte 4 (galeria de 3 trechos de falha), pedimos pra IA estender a
figura estática pra um vídeo do trecho inteiro, mostrando a caixa prevista derivando durante
a oclusão quadro a quadro. Reaproveitou o desenho de caixas que já existia
(`_retangulo`/`_enquadrar`) e `imageio.mimwrite` (mesma função já usada por
`inferencia.ipynb`) — sem dependência nova. Fizemos o mesmo pedido pra Parte 5 (vídeo do
vídeo subsamostrado a 1/2 e 1/5 da taxa original, com as caixas previstas), reaproveitando
dessa vez `desenhar()`/`cor_da_identidade()` de `src/inference/predict.py`.

### 4. Revisão de outra sessão de Claude — achou 3 problemas reais

Antes de uma apresentação, rodamos a mesma tarefa numa segunda instância do Claude (sessão
separada, sem o contexto desta conversa) pra auditar o estado do repositório. Ela encontrou:

1. **Config fora de sincronia com o checkpoint**: o `best.pth` tinha sido retreinado pelo
   grid search com `lr=0,01, batch_size=64`, mas `configs/mot17_gru.yaml` ainda dizia
   `lr=0,001, batch_size=256` — rodar `make treinar` não reproduziria o modelo apresentado.
2. **A "vencedora" do grid search era ruído**: as 5 melhores combinações tinham IDF1 dentro
   do desvio-padrão umas das outras (ex.: diferença de 0,0030 contra desvio de 0,0049) — a
   mesma régua que a Parte 3 já usa ("efeito só conta se for maior que o espalhamento entre
   seeds") não tinha sido aplicada à escolha do hiperparâmetro.
3. **Ablação (Parte 3) e modelo final em regimes de treino diferentes**: a ablação comparando
   RNN/LSTM/GRU foi treinada com `lr=0,001`; o modelo de produção usa `lr=0,01` — os números
   de uma não são comparáveis com os da outra sem essa ressalva.

Corrigimos os três: sincronizamos o YAML com o checkpoint; reescrevemos a geração do
`report.md` do grid search (`src/gridsearch/runner.py`) pra detectar e marcar
explicitamente empates estatísticos em vez de só apontar uma "vencedora"; e documentamos a
diferença de regime no README. Também ajustamos o `.gitignore` (os checkpoints intermediários
do grid search, 24 MB, tinham entrado no git sem querer) e tiramos eles do índice.

### 5. O horizonte de memória "morria" só por falta de régua

Medindo o horizonte analítico de memória (norma do gradiente) do modelo retreinado, o número
batia na borda da janela medida sem nunca cruzar o corte de 1% — ou seja, a função só estava
dizendo "onde a régua acabou", não "onde o gradiente morreu de verdade". Pedimos pra rodar a
mesma medida com uma janela maior (T=32, sem precisar retreinar nada — é só uma medição em
cima do modelo já treinado) e descobrimos que o gradiente **não desaparece** nesse regime de
`lr`: ele cai nos primeiros passos e platô em ~65% do valor original. Isso expôs uma legenda
enganosa que o próprio código gerava ("o gradiente morre em N passos" mesmo quando não
morreu) — corrigimos `src/fails/runner.py` pra distinguir os dois casos e escrever a legenda
certa em cada um.

## O que aprendemos sobre usar IA neste trabalho

- **IA implementa bem em cima de uma decisão já discutida**, mas a parte de "isso é
  estatisticamente defensável?" (o caso do grid search) só apareceu depois de pedirmos
  explicitamente, ou numa segunda revisão independente — vale pedir isso de propósito, não
  assumir que vem de graça.
- **Uma segunda sessão sem o contexto da primeira pegou coisas que a gente (e a primeira
  sessão) não tinha visto** — os três problemas do episódio 4 só vieram à tona porque
  pedimos uma auditoria fresca, sem o viés de quem construiu o código.
- **Reprodutibilidade exige checar, não supor**: o config e o checkpoint saíram de sincronia
  mais de uma vez durante o trabalho (treino manual vs. retreino do grid search), e o
  enunciado exige exatamente o contrário ("toda tabela e curva tem que ser reproduzível a
  partir do repositório").
- **Na apresentação, a imagem é a única fonte de verdade que o avaliador tem** — uma legenda
  gerada automaticamente mas desatualizada (episódio 5) é pior que não ter a medida, porque
  ela afirma algo específico e errado. Isso só foi pego relendo os números por trás da
  figura, não olhando a figura pronta.
