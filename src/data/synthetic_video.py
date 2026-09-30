"""Gerador de vídeos sintéticos — artefato 1 da Parte 0.

Vídeos ``size x size`` de 30 a 60 quadros, com 5 a 15 elipses em movimento, tamanhos
variados, ruído e contraste variáveis. É o ambiente controlado onde se sabe a resposta
certa: a identidade de cada objeto é conhecida por construção, então dá para depurar a
associação e as métricas inteiras antes de encostar no MOT17.

Herda do gerador de imagens do PA1 (``semantic-segmentation/src/data/synthetic.py``) o
desenho da elipse, o ruído gaussiano e a variação de ganho/brilho. O que muda: a elipse
deixa de ser um carimbo e vira um objeto com **estado persistente** — id, posição,
velocidade, cor e profundidade — simulado ao longo dos quadros.

Dois requisitos do enunciado moldam o desenho deste módulo:

**A oclusão precisa ser real, não aparente.**
    As elipses são desenhadas em ordem de profundidade, e a visibilidade de cada objeto é
    medida comparando a máscara que ele teria sozinho com o que sobra dela depois de tudo
    que está na frente. Um objeto que passa atrás de outro realmente some.

**A duração da oclusão precisa ser um botão.**
    Deixar isso para o acaso dos cruzamentos não dá controle nenhum sobre a duração — e a
    Parte 4 pede justamente para comparar o horizonte de memória do modelo com a
    *distribuição de duração de oclusão do dataset*. Por isso existe o **occluder
    calibrado**: uma barra vertical fixa, tipo um poste, com largura calculada a partir da
    velocidade para que um objeto designado atravesse escondido por exatamente
    ``occlusion_duration`` quadros. Ver ``_occluder_calibrado``.

Oclusão mútua entre elipses continua acontecendo naturalmente pela ordem de profundidade —
as duas convivem, e só a do poste é previsível.
"""

import colorsys
from dataclasses import dataclass, field

import numpy as np
from skimage.draw import ellipse

from src.data.sequence import Frame, Sequence


@dataclass
class MovingEllipse:
    """Uma elipse com estado que persiste entre quadros.

    Attributes:
        id: identidade, estável do nascimento ao fim da sequência.
        center: (linha, coluna) em pixels, subpixel.
        velocity: (d_linha, d_coluna) em pixels por quadro.
        r_radius, c_radius: semi-eixos em pixels.
        rotation: rotação da elipse em radianos.
        color: (3,) RGB em [0, 1]. Própria e **estável** ao longo do vídeo — é o que
            permite testar uma memória de aparência (Trilha B) já no sintético.
        z: profundidade. Maior = mais perto da câmera = desenhado por cima.
    """

    id: int
    center: np.ndarray
    velocity: np.ndarray
    r_radius: int
    c_radius: int
    rotation: float
    color: np.ndarray
    z: int

    def step(self, size: int) -> None:
        """Avança um quadro, quicando nas bordas.

        Quicar (em vez de deixar sair de cena) é deliberado: mantém as trajetórias longas,
        e nesta parte o que interessa medir é a identidade sob oclusão, não sob nascimento
        e morte de tracks — isso vem depois, com o MOT17.
        """
        self.center = self.center + self.velocity
        for eixo, raio in ((0, self.r_radius), (1, self.c_radius)):
            minimo, maximo = raio, size - 1 - raio
            if self.center[eixo] < minimo:
                self.center[eixo] = 2 * minimo - self.center[eixo]
                self.velocity[eixo] *= -1
            elif self.center[eixo] > maximo:
                self.center[eixo] = 2 * maximo - self.center[eixo]
                self.velocity[eixo] *= -1

    def mask_at(self, center: np.ndarray, size: int) -> np.ndarray:
        """Máscara booleana da elipse posta num centro qualquer, sem nada na frente."""
        rr, cc = ellipse(
            center[0], center[1], self.r_radius, self.c_radius,
            rotation=self.rotation, shape=(size, size),
        )
        m = np.zeros((size, size), dtype=bool)
        m[rr, cc] = True
        return m

    def mask(self, size: int) -> np.ndarray:
        """Máscara booleana que a elipse teria **sozinha**, sem nada na frente."""
        return self.mask_at(self.center, size)

    def half_width(self, size: int) -> float:
        """Semi-extensão horizontal do desenho, em pixels.

        Não é ``c_radius``: a elipse é rotacionada, e depois da rotação a largura do que
        aparece na tela é uma mistura dos dois semi-eixos. Usar ``c_radius`` para calibrar
        o poste erra a duração da oclusão em vários quadros — foi o que aconteceu na
        primeira versão (pediu 10, entregou 14).

        A medição é feita no centro do quadro para não sofrer corte de borda, e é a
        extensão **contínua** (``max - min``, sem o ``+1`` da contagem de pixels): é a
        largura que entra numa conta de geometria, não o número de colunas acesas.
        """
        centro = np.array([size / 2.0, size / 2.0])
        colunas = np.flatnonzero(self.mask_at(centro, size).any(axis=0))
        return (colunas.max() - colunas.min()) / 2.0


@dataclass
class Occluder:
    """Barra vertical estática que esconde tudo que passa atrás dela."""

    left: int
    width: int
    color: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.08, 0.10]))

    def mask(self, size: int) -> np.ndarray:
        m = np.zeros((size, size), dtype=bool)
        m[:, self.left:self.left + self.width] = True
        return m


class SyntheticVideos:
    """Conjunto de vídeos sintéticos, pré-gerados em memória.

    Mesmo padrão do ``SyntheticEllipses`` do PA1: tudo é gerado no ``__init__`` a partir da
    seed, de modo que a mesma seed sempre produz os mesmos vídeos. ``__getitem__`` devolve
    uma ``Sequence`` — não um tensor: quadros de uma sequência não são amostras
    independentes, e o que a Parte 2 vai treinar são janelas recortadas destas sequências,
    não as sequências em si.

    Por isso esta classe **não** herda de ``torch.utils.data.Dataset``: ela nunca vai para
    dentro de um DataLoader. O Dataset de verdade aparece na Parte 2, sobre as janelas.

    Args:
        n_sequences: quantos vídeos gerar.
        size: lado do quadro em pixels.
        seed: semente; a mesma sempre produz os mesmos vídeos.
        min_frames, max_frames: faixa de comprimento dos vídeos, em quadros.
        min_obj, max_obj: faixa do número de elipses por vídeo.
        speed: velocidade típica em pixels por quadro. Cada objeto sorteia em torno dela.
        occlusion_duration: quantos quadros o objeto designado fica **totalmente** escondido
            atrás do poste. É o botão que a Parte 1 vai girar.
        n_occluders: quantas barras. A primeira é a calibrada; as demais são aleatórias.
        noise: desvio do ruído gaussiano por quadro.
        fps: metadado da sequência; usado pela Parte 5 (queda de taxa de quadros).
        prefix: prefixo do nome das sequências (``SYNTH-00``, ``SYNTH-01``, ...).
    """

    def __init__(
        self,
        n_sequences: int,
        size: int = 128,
        seed: int = 42,
        min_frames: int = 30,
        max_frames: int = 60,
        min_obj: int = 5,
        max_obj: int = 15,
        speed: float = 2.0,
        occlusion_duration: int = 10,
        n_occluders: int = 1,
        noise: float = 0.05,
        fps: float = 30.0,
        prefix: str = "SYNTH",
    ):
        self.n_sequences = n_sequences
        self.size = size
        self.seed = seed
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.min_obj = min_obj
        self.max_obj = max_obj
        self.speed = speed
        self.occlusion_duration = occlusion_duration
        self.n_occluders = n_occluders
        self.noise = noise
        self.fps = fps
        self.prefix = prefix

        # uma seed por sequência: gerar a sequência 3 sozinha dá o mesmo resultado que
        # gerar as 10 e pegar a terceira. Facilita reproduzir um caso isolado.
        self._sequences = [
            self._gerar(indice) for indice in range(n_sequences)
        ]

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, indice: int) -> Sequence:
        return self._sequences[indice]

    # ------------------------------------------------------------------ geração

    def _gerar(self, indice: int) -> Sequence:
        rng = np.random.RandomState(self.seed + 1000 * indice)

        n_frames = int(rng.randint(self.min_frames, self.max_frames + 1))
        n_obj = int(rng.randint(self.min_obj, self.max_obj + 1))

        objetos = self._criar_objetos(rng, n_obj, n_frames)
        occluders = self._criar_occluders(rng, objetos[0])

        # ganho e brilho são propriedade da "câmera": sorteados uma vez por vídeo, não por
        # quadro. Sorteá-los por quadro produziria uma cintilação que não existe em vídeo
        # real e que tornaria a aparência inútil como pista de identidade.
        gain = rng.uniform(0.8, 1.2)
        offset = rng.uniform(-0.05, 0.05)
        bg = rng.uniform(0.0, 0.15)

        frames = []
        for t in range(n_frames):
            frames.append(self._render(objetos, occluders, t, rng, bg, gain, offset))
            for obj in objetos:
                obj.step(self.size)

        return Sequence(
            name=f"{self.prefix}-{indice:02d}",
            fps=self.fps,
            width=self.size,
            height=self.size,
            frames=frames,
        )

    def _criar_objetos(self, rng, n_obj: int, n_frames: int) -> list[MovingEllipse]:
        """Cria as elipses. A de id 1 é a designada para a oclusão calibrada."""
        # profundidades distintas: com z repetido a ordem de desenho seria arbitrária e a
        # visibilidade calculada não corresponderia ao que a imagem mostra.
        profundidades = rng.permutation(n_obj)
        # matizes espaçados no círculo de cores, embaralhados: duas elipses vizinhas na
        # lista não saem com cores parecidas, e a aparência vira uma pista útil.
        matizes = rng.permutation(np.linspace(0, 1, n_obj, endpoint=False))

        objetos = []
        for i in range(n_obj):
            r_radius = int(rng.randint(5, 15))
            c_radius = int(rng.randint(5, 15))
            cor = np.array(colorsys.hsv_to_rgb(
                matizes[i], rng.uniform(0.6, 1.0), rng.uniform(0.6, 0.95)
            ))

            obj = MovingEllipse(
                id=i + 1,
                center=np.zeros(2),
                velocity=np.zeros(2),
                r_radius=r_radius,
                c_radius=c_radius,
                rotation=rng.uniform(0, 2 * np.pi),
                color=cor,
                z=int(profundidades[i]),
            )

            if i == 0:
                # o objeto da oclusão calibrada: velocidade puramente horizontal e de
                # módulo exatamente `speed`, para que a largura do poste se traduza em
                # uma duração previsível. Ver _largura_calibrada.
                obj.center, obj.velocity = self._trajetoria_calibrada(rng, obj, n_frames)
            else:
                obj.center = np.array([
                    rng.uniform(r_radius, self.size - 1 - r_radius),
                    rng.uniform(c_radius, self.size - 1 - c_radius),
                ])
                angulo = rng.uniform(0, 2 * np.pi)
                modulo = self.speed * rng.uniform(0.5, 1.5)
                obj.velocity = np.array([modulo * np.sin(angulo), modulo * np.cos(angulo)])

            objetos.append(obj)
        return objetos

    def _trajetoria_calibrada(self, rng, obj: MovingEllipse, n_frames: int):
        """Posição e velocidade iniciais do objeto que atravessa o poste.

        Ele entra na oclusão total por volta de 25% do vídeo, o que deixa quadros
        suficientes antes (para a track existir) e depois (para ela voltar) — que é
        exatamente a figura pedida no enunciado: some por N quadros e volta.
        """
        meia_largura = obj.half_width(self.size)
        largura = self._largura_calibrada(meia_largura)
        esquerda = (self.size - largura) // 2

        # coluna em que o objeto fica completamente escondido pela primeira vez: o seu
        # lado esquerdo acabou de passar da borda esquerda do poste
        entrada = esquerda + meia_largura
        recuo = 0.25 * n_frames * self.speed
        limite = self.size - 1 - obj.c_radius
        coluna = float(np.clip(entrada - recuo, obj.c_radius, limite))

        linha = rng.uniform(obj.r_radius, self.size - 1 - obj.r_radius)
        return np.array([linha, coluna]), np.array([0.0, float(self.speed)])

    def _largura_calibrada(self, meia_largura: float) -> int:
        """Largura do poste que esconde por ``occlusion_duration`` quadros.

        O objeto está totalmente escondido enquanto o desenho inteiro cabe dentro da
        barra. Com uma barra de largura ``W``, isso vale por
        ``(W - 2*meia_largura) / speed`` quadros, de onde sai
        ``W = occlusion_duration * speed + 2*meia_largura``.

        ``meia_largura`` é a extensão horizontal do desenho já rotacionado
        (``MovingEllipse.half_width``), não ``c_radius`` — ver a docstring de lá.

        A largura é limitada a 80% do quadro: um poste maior que isso não deixaria espaço
        para o objeto aparecer antes e depois, e a figura perderia o sentido. Quando o
        limite morde, a duração obtida é menor que a pedida — por isso existe
        ``duracao_esperada``, que devolve o que de fato vai acontecer.

        Medido: a duração observada bate com a pedida dentro de ±1 quadro para
        ``occlusion_duration`` de 5 a 34 e ``speed`` de 1 a 4. O resíduo é a diferença
        entre um intervalo contínuo e a contagem de quadros inteiros dentro dele.
        """
        largura = int(round(self.occlusion_duration * self.speed + 2 * meia_largura))
        return int(min(largura, 0.8 * self.size))

    def duracao_esperada(self, obj: MovingEllipse) -> float:
        """Quantos quadros a oclusão calibrada de fato dura, para um dado objeto."""
        meia_largura = obj.half_width(self.size)
        largura = self._largura_calibrada(meia_largura)
        return max(0.0, (largura - 2 * meia_largura) / self.speed)

    def _criar_occluders(self, rng, designado: MovingEllipse) -> list[Occluder]:
        """O poste calibrado, mais as barras extras pedidas em ``n_occluders``."""
        largura = self._largura_calibrada(designado.half_width(self.size))
        occluders = [Occluder(left=(self.size - largura) // 2, width=largura)]

        for _ in range(max(0, self.n_occluders - 1)):
            w = int(rng.randint(4, max(5, self.size // 12)))
            occluders.append(Occluder(left=int(rng.randint(0, self.size - w)), width=w))
        return occluders

    # ------------------------------------------------------------------ render

    def _render(self, objetos, occluders, t: int, rng, bg, gain, offset) -> Frame:
        """Desenha um quadro e mede a visibilidade real de cada objeto.

        A ordem importa e é a mesma nos dois: do mais distante para o mais próximo, e o
        poste por último. O que a imagem mostra e o que ``visibility`` afirma vêm da mesma
        ordenação, então não há como um dizer uma coisa e o outro dizer outra.
        """
        size = self.size
        imagem = np.full((size, size, 3), bg, dtype=np.float32)

        mascaras = {obj.id: obj.mask(size) for obj in objetos}
        occluder_mask = np.zeros((size, size), dtype=bool)
        for occ in occluders:
            occluder_mask |= occ.mask(size)

        do_fundo_para_frente = sorted(objetos, key=lambda o: o.z)
        for obj in do_fundo_para_frente:
            imagem[mascaras[obj.id]] = obj.color
        for occ in occluders:
            imagem[occ.mask(size)] = occ.color

        boxes, ids, visibilidades = [], [], []
        for obj in objetos:
            isolada = mascaras[obj.id]
            area = int(isolada.sum())
            if area == 0:
                # a elipse degenerou (não deve acontecer com o quique, mas se acontecer é
                # melhor omitir do que anotar uma caixa vazia)
                continue

            # tudo que está na frente: os objetos de z maior, mais o poste, que é sempre
            # o mais próximo da câmera
            na_frente = occluder_mask.copy()
            for outro in objetos:
                if outro.z > obj.z:
                    na_frente |= mascaras[outro.id]

            visivel = isolada & ~na_frente
            visibilidades.append(visivel.sum() / area)

            # caixa AMODAL: o retângulo do objeto inteiro, inclusive da parte escondida.
            # É a convenção do MOT17, e é o que dá sentido a `visibility` — a caixa diz
            # onde o objeto está, e a visibilidade diz quanto dele se vê.
            linhas, colunas = np.nonzero(isolada)
            boxes.append([colunas.min(), linhas.min(), colunas.max() + 1, linhas.max() + 1])
            ids.append(obj.id)

        # ruído por quadro (é da captura, varia a cada quadro); ganho e brilho são da
        # câmera e já vieram fixos da sequência
        imagem = imagem + rng.normal(0, self.noise, imagem.shape).astype(np.float32)
        imagem = np.clip(imagem * gain + offset, 0.0, 1.0)

        return Frame(
            index=t,
            image=(imagem * 255).astype(np.uint8),
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            ids=np.array(ids, dtype=np.int64),
            visibility=np.array(visibilidades, dtype=np.float32),
        )
