"""Métricas de identidade ao longo do tempo — o artefato 3 da Parte 0.

O enunciado proíbe ``motmetrics``, ``TrackEval`` e ``py-motmetrics``: **nós implementamos
IDF1 e a contagem de ID switches**. Este módulo é essa implementação, e é a peça mais
importante do projeto — se ela estiver errada, todo número da apresentação está errado,
inclusive os que parecem bons. Por isso ela vem antes do modelo, e por isso tem os três
testes construídos à mão que o enunciado exige (``tests/test_metrics.py``).

## As duas perguntas são diferentes, e por isso são duas métricas

**IDF1 pergunta: os rótulos estão certos?** É uma atribuição **global**, um-para-um, entre
as identidades previstas e as verdadeiras, decidida uma vez sobre a sequência inteira. Cada
identidade verdadeira ganha *uma* identidade prevista como sua representante oficial, e tudo
que essa representante fez fora do lugar certo conta contra. É por isso que ela pune de
verdade o rastreador que parte uma pessoa em três tracks: duas das três serão falso
positivo do começo ao fim, por mais que casem perfeitamente quadro a quadro.

**ID switches pergunta: quantas vezes o rótulo mudou?** É uma contagem **local**, quadro a
quadro, de eventos. Um rastreador pode ter poucos switches e IDF1 baixo (partiu tudo e
nunca trocou), ou muitos switches e IDF1 razoável (trocou e voltou). As duas juntas dizem o
que uma sozinha não diz — que é exatamente por que o enunciado pede as duas.

O caso (c) dos testes obrigatórios existe para provar que a implementação distingue as duas
coisas: uma track partida no meio tem efeito no IDF1 e **zero** switches.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.sequence import Sequence
from src.metrics.iou import iou_matrix
from src.metrics.matching import match_clearmot

#: IoU mínima para uma caixa prevista poder representar uma verdadeira. 0,5 é a convenção do
#: MOTChallenge; está aqui como constante nomeada porque o número entra em três lugares e
#: mudá-lo em um só daria métricas que não conversam entre si.
LIMIAR_IOU = 0.5


@dataclass
class IDF1Result:
    """Resultado da atribuição global de identidades."""

    idf1: float
    idp: float          #: precisão de identidade — IDTP / (IDTP + IDFP)
    idr: float          #: revocação de identidade — IDTP / (IDTP + IDFN)
    idtp: int           #: quadros-objeto em que a representante oficial estava certa
    idfp: int
    idfn: int
    #: identidade verdadeira → identidade prevista escolhida como representante oficial.
    #: É o que as figuras da Parte 4 usam para colorir predição e gabarito de forma
    #: consistente: sem isso, as cores dos dois painéis não teriam relação nenhuma.
    correspondencia: dict[int, int] = field(default_factory=dict)


@dataclass
class TrackingMetrics:
    """Tudo que se mede de um rastreamento, num objeto só."""

    idf1: float
    idp: float
    idr: float
    id_switches: int
    fragmentations: int
    n_gt_ids: int
    n_pred_ids: int
    count_error: int        #: |previstas − verdadeiras|, o análogo temporal do PA1
    mota: float
    false_positives: int
    false_negatives: int
    n_gt_boxes: int
    idtp: int
    idfp: int
    idfn: int
    correspondencia: dict[int, int] = field(default_factory=dict)
    #: por quadro, para as figuras: casados, FP e FN. Barato de guardar e caro de refazer.
    por_quadro: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Versão serializável — a correspondência e o por-quadro ficam de fora do resumo."""
        return {
            "idf1": self.idf1, "idp": self.idp, "idr": self.idr,
            "id_switches": self.id_switches, "fragmentations": self.fragmentations,
            "n_gt_ids": self.n_gt_ids, "n_pred_ids": self.n_pred_ids,
            "count_error": self.count_error, "mota": self.mota,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "n_gt_boxes": self.n_gt_boxes,
            "idtp": self.idtp, "idfp": self.idfp, "idfn": self.idfn,
        }


def _conferir_alinhamento(gt: Sequence, pred: Sequence) -> None:
    if len(gt) != len(pred):
        raise ValueError(
            f"gabarito tem {len(gt)} quadros e a predição tem {len(pred)}. As duas "
            f"sequências precisam cobrir os mesmos quadros — comparar comprimentos "
            f"diferentes daria um IDF1 que parece bom só porque metade do vídeo sumiu."
        )


def compute_idf1(
    gt: Sequence, pred: Sequence, threshold: float = LIMIAR_IOU
) -> IDF1Result:
    """IDF1 por atribuição global um-para-um sobre a sequência inteira.

    O algoritmo, em três passos:

    1. **contar coexistências.** Para cada par (identidade verdadeira *i*, identidade
       prevista *j*), quantos quadros existem em que as duas aparecem e as caixas casam com
       IoU ≥ ``threshold``. Esse é ``overlap[i, j]``;
    2. **escolher as representantes.** Custo de casar *i* com *j* é
       ``|i| + |j| − 2·overlap[i, j]``: os quadros de *i* que *j* não cobriu (falso negativo
       de identidade) mais os de *j* que não eram de *i* (falso positivo). O Hungarian
       escolhe a atribuição de custo mínimo. A matriz é quadrada e com preenchimento, para
       que **não casar** também seja uma opção com preço: deixar *i* sem representante custa
       ``|i|``, todos os seus quadros virando falso negativo;
    3. **somar.** ``IDTP`` é o total de coexistências das duplas escolhidas; o resto de
       cada lado é ``IDFN`` e ``IDFP``.

    Returns:
        ``IDF1Result``, com a correspondência escolhida — que as figuras reaproveitam para
        colorir os dois painéis de forma consistente.
    """
    _conferir_alinhamento(gt, pred)

    ids_gt, ids_pred = gt.track_ids, pred.track_ids
    n, m = len(ids_gt), len(ids_pred)

    if n == 0 and m == 0:
        # nada para acertar e nada errado: 1,0. Devolver 0,0 aqui faria uma sequência vazia
        # parecer um fracasso do rastreador, e faria a média entre sequências mentir.
        return IDF1Result(1.0, 1.0, 1.0, 0, 0, 0, {})
    if n == 0 or m == 0:
        # um lado vazio e o outro não: nada casa, e tudo que existe do lado cheio é erro
        return IDF1Result(
            idf1=0.0, idp=0.0, idr=0.0, idtp=0,
            idfp=_total_caixas(pred), idfn=_total_caixas(gt), correspondencia={},
        )

    onde_gt = {int(g): i for i, g in enumerate(ids_gt)}
    onde_pred = {int(p): j for j, p in enumerate(ids_pred)}

    overlap = np.zeros((n, m), dtype=np.int64)
    tamanho_gt = np.zeros(n, dtype=np.int64)
    tamanho_pred = np.zeros(m, dtype=np.int64)

    for quadro_gt, quadro_pred in zip(gt, pred):
        linhas = [onde_gt[int(g)] for g in quadro_gt.ids]
        colunas = [onde_pred[int(p)] for p in quadro_pred.ids]
        tamanho_gt[linhas] += 1
        tamanho_pred[colunas] += 1
        if linhas and colunas:
            casam = iou_matrix(quadro_gt.boxes, quadro_pred.boxes) >= threshold
            overlap[np.ix_(linhas, colunas)] += casam

    # ---- matriz quadrada com preenchimento -------------------------------------------
    # bloco superior-esquerdo: casar i com j.  superior-direito: deixar i sem representante.
    # inferior-esquerdo: deixar j sem representada.  inferior-direito: zero, é o descarte.
    total = int(tamanho_gt.sum() + tamanho_pred.sum())
    proibido = float(total + 1)  # maior que qualquer atribuição viável, e finito

    custo = np.full((n + m, n + m), proibido, dtype=np.float64)
    custo[:n, :m] = tamanho_gt[:, None] + tamanho_pred[None, :] - 2 * overlap
    np.fill_diagonal(custo[:n, m:], tamanho_gt)
    np.fill_diagonal(custo[n:, :m], tamanho_pred)
    custo[n:, m:] = 0.0

    linhas, colunas = linear_sum_assignment(custo)

    idtp = 0
    correspondencia: dict[int, int] = {}
    for i, j in zip(linhas, colunas):
        if i < n and j < m:
            idtp += int(overlap[i, j])
            correspondencia[int(ids_gt[i])] = int(ids_pred[j])

    idfn = int(tamanho_gt.sum()) - idtp
    idfp = int(tamanho_pred.sum()) - idtp

    # IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN), que se simplifica para 2·IDTP dividido pelo
    # total de caixas dos dois lados — já que IDFP + IDFN = |pred| + |gt| − 2·IDTP
    denominador = 2 * idtp + idfp + idfn
    return IDF1Result(
        idf1=(2 * idtp / denominador) if denominador else 1.0,
        idp=(idtp / (idtp + idfp)) if (idtp + idfp) else 0.0,
        idr=(idtp / (idtp + idfn)) if (idtp + idfn) else 0.0,
        idtp=idtp, idfp=idfp, idfn=idfn, correspondencia=correspondencia,
    )


def compute_clearmot(
    gt: Sequence, pred: Sequence, threshold: float = LIMIAR_IOU
) -> dict:
    """ID switches, fragmentações, FP, FN e MOTA — a passada quadro a quadro.

    Cada quadro é casado com ``match_clearmot``, que **preserva o par do quadro anterior**
    enquanto ele valer (ver ``src/metrics/matching.py``). É o que faz a contagem medir o
    rastreador em vez de medir a própria métrica.

    Duas definições que o enunciado manda contar explicitamente, e que implementações
    diferentes contam de formas diferentes — aqui ficam escritas:

    **ID switch.** Uma identidade verdadeira que estava casada com a previsão *A* passa a
    estar casada com *B* ≠ *A*. A memória do último par **atravessa buracos**: se o objeto
    some por 30 quadros e volta com outro id, isso é um switch. É o comportamento que
    interessa ao PA2 inteiro — é literalmente o enunciado ("se um objeto aparece no quadro 3
    e reaparece no quadro 40, ele tem que sair com o mesmo identificador").

    **Fragmentação.** Uma identidade verdadeira que já foi rastreada deixa de ser e **volta
    a ser**. Contamos retomadas, não interrupções: uma interrupção que nunca se recupera é
    indistinguível da track simplesmente acabando, e contá-la castigaria o rastreador por
    um objeto que saiu de cena. Consequência a assumir: o número aqui é menor do que o de
    implementações que contam toda interrupção.
    """
    _conferir_alinhamento(gt, pred)

    ultimo_par: dict[int, int] = {}       # identidade verdadeira → última prevista
    ja_rastreado: set[int] = set()        # já teve casamento alguma vez
    perdido_desde: set[int] = set()       # estava rastreada, deixou de estar
    id_switches = 0
    fragmentations = 0
    fp_total = 0
    fn_total = 0
    n_gt_boxes = 0
    por_quadro: list[dict] = []

    for quadro_gt, quadro_pred in zip(gt, pred):
        iou = iou_matrix(quadro_gt.boxes, quadro_pred.boxes)
        pares = match_clearmot(iou, quadro_gt.ids, quadro_pred.ids, ultimo_par, threshold)

        casados_gt = set()
        switches_aqui = 0
        for i, j in pares:
            g, p = int(quadro_gt.ids[i]), int(quadro_pred.ids[j])
            casados_gt.add(g)

            anterior = ultimo_par.get(g)
            if anterior is not None and anterior != p:
                id_switches += 1
                switches_aqui += 1
            ultimo_par[g] = p

            if g in perdido_desde:
                fragmentations += 1
                perdido_desde.discard(g)
            ja_rastreado.add(g)

        # uma identidade que já foi rastreada e **está no quadro** sem casar está perdida
        # agora. Objeto ausente do gabarito não conta: ele não está perdido, ele não está.
        for g in quadro_gt.ids:
            g = int(g)
            if g in ja_rastreado and g not in casados_gt:
                perdido_desde.add(g)

        fp = len(quadro_pred) - len(pares)
        fn = len(quadro_gt) - len(pares)
        fp_total += fp
        fn_total += fn
        n_gt_boxes += len(quadro_gt)
        por_quadro.append({
            "frame": quadro_gt.index, "matches": len(pares),
            "fp": fp, "fn": fn, "id_switches": switches_aqui,
            "n_gt": len(quadro_gt), "n_pred": len(quadro_pred),
        })

    mota = (
        1.0 - (fn_total + fp_total + id_switches) / n_gt_boxes if n_gt_boxes else 1.0
    )
    return {
        "id_switches": id_switches, "fragmentations": fragmentations,
        "false_positives": fp_total, "false_negatives": fn_total,
        "n_gt_boxes": n_gt_boxes, "mota": mota, "por_quadro": por_quadro,
    }


def evaluate(
    gt: Sequence, pred: Sequence, threshold: float = LIMIAR_IOU
) -> TrackingMetrics:
    """Todas as métricas de uma sequência. É esta a função que o resto do projeto chama.

    Args:
        gt: sequência do gabarito, com as identidades verdadeiras.
        pred: sequência prevista pelo rastreador, com os mesmos quadros.
        threshold: IoU mínima para uma caixa prevista representar uma verdadeira.
    """
    ident = compute_idf1(gt, pred, threshold)
    clear = compute_clearmot(gt, pred, threshold)

    n_gt_ids = len(gt.track_ids)
    n_pred_ids = len(pred.track_ids)

    return TrackingMetrics(
        idf1=ident.idf1, idp=ident.idp, idr=ident.idr,
        idtp=ident.idtp, idfp=ident.idfp, idfn=ident.idfn,
        correspondencia=ident.correspondencia,
        id_switches=clear["id_switches"], fragmentations=clear["fragmentations"],
        n_gt_ids=n_gt_ids, n_pred_ids=n_pred_ids,
        count_error=abs(n_pred_ids - n_gt_ids),
        mota=clear["mota"],
        false_positives=clear["false_positives"],
        false_negatives=clear["false_negatives"],
        n_gt_boxes=clear["n_gt_boxes"],
        por_quadro=clear["por_quadro"],
    )


def _total_caixas(sequence: Sequence) -> int:
    return int(sum(len(f) for f in sequence))
