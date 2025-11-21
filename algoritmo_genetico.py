# algoritmo_genetico_revisado.py
"""
Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)
Versão revisada: docstrings completas + coleta explícita da média da população

Dependências:
 - utils.calcular_matriz_distancias
 - utils.criar_rota_aleatoria
 - utils.calcular_distancia_total

Formato de retorno: a classe AlgoritmoGenetico.executar() retorna o Melhor Individuo (objeto Individuo)

Autor: Gerado por assistente (revisão solicitada pelo usuário)
"""

from typing import List, Tuple, Optional
import random
import utils

# ======================================================================
# Classe Individuo
# ======================================================================
class Individuo:
    """
    Representa uma solução do TSP (uma rota completa).

    Attributes:
        rota (List[int]): Sequência de índices das cidades.
        cidades (List[Tuple[float, float]]): Lista de coordenadas (x, y).
        dist_matrix (Optional[List[List[float]]]): Matriz pré-calculada de distâncias.
        distancia (float): Distância total da rota.
        fitness (float): Inverso da distância (maior = melhor).
    """

    # pequena constante para evitar divisão por zero ao calcular fitness
    DEFAULT_EPSILON = 1e-9

    def __init__(self, rota: List[int], cidades: List[Tuple[float, float]],
                 dist_matrix: Optional[List[List[float]]] = None,
                 epsilon: float = DEFAULT_EPSILON):
        """
        Inicializa o indivíduo e calcula distância e fitness.

        Args:
            rota: permutação representando a rota.
            cidades: lista de coordenadas.
            dist_matrix: matriz de distâncias opcional (ganho de performance).
            epsilon: pequena constante para evitar divisão por zero.
        """
        # Guardamos uma cópia da rota para evitar aliasing acidental
        self.rota: List[int] = rota[:]
        self.cidades = cidades
        self.dist_matrix = dist_matrix
        self.epsilon = epsilon

        # Calcula a distância total (usa matriz se fornecida)
        # PARA APRESENTAÇÃO: "Aqui transformamos a rota em seu custo (distância total)."
        self.distancia: float = utils.calcular_distancia_total(self.rota, self.cidades, self.dist_matrix)

        # Fitness: transformamos minimização (distância) em maximização (fitness).
        # PARA APRESENTAÇÃO: "Fitness = 1 / (distância + epsilon) para evitar divisão por zero."
        self.fitness: float = 1.0 / (self.distancia + self.epsilon)

    def copy(self) -> "Individuo":
        """Retorna um clone independente deste indivíduo."""
        # PARA APRESENTAÇÃO: "Usamos cópias para garantir que o elitismo não compartilhe referências."
        return Individuo(self.rota[:], self.cidades, self.dist_matrix, self.epsilon)

    def __repr__(self):
        # Representação curta útil no console
        return f"[Indivíduo: Dist={self.distancia:.4f} | Rota={self.rota[:6]}...]"

# ======================================================================
# Classe AlgoritmoGenetico
# ======================================================================
class AlgoritmoGenetico:
    """
    Algoritmo Genético completo para TSP.

    Funcionalidades:
      - seleção: torneio, roleta, ranking
      - crossover: OX, PMX
      - mutação: swap, inversion
      - elitismo
      - suporte a matriz de distâncias pré-calculada
      - coleta de histórico: melhor por geração e média da população
    """

    # Constantes de segurança / limites
    MAX_SELECTION_ATTEMPTS = 5
    MAX_POPULATION_FILL_ATTEMPTS_MULTIPLIER = 10
    MIN_TORNEIO_SIZE = 2
    MAX_ELITISMO = 0.5
    MIN_POPULACAO = 2
    MIN_GERACOES = 1

    def __init__(
        self,
        cidades: List[Tuple[float, float]],
        tamanho_populacao: int,
        num_geracoes: int,
        taxa_mutacao: float,
        taxa_elitismo: float = 0.05,
        metodo_selecao: str = "torneio",
        metodo_crossover: str = "ox",
        metodo_mutacao: str = "swap",
        torneio_k: int = 3,
        dist_matrix: Optional[List[List[float]]] = None,
        rng_seed: Optional[int] = None,
        epsilon: float = Individuo.DEFAULT_EPSILON,
    ):
        """
        Inicializa o AG com parâmetros básicos e validações.

        Args:
            cidades: lista de coordenadas (x, y).
            tamanho_populacao: tamanho da população.
            num_geracoes: número de gerações para executar.
            taxa_mutacao: probabilidade (0..1) de aplicar mutação.
            taxa_elitismo: fração (0..MAX_ELITISMO) de população preservada como elite.
            metodo_selecao: 'torneio'|'roleta'|'ranking'.
            metodo_crossover: 'ox'|'pmx'.
            metodo_mutacao: 'swap'|'inversion'.
            torneio_k: tamanho do torneio (para seleção por torneio).
            dist_matrix: matriz de distâncias pré-calculada (opcional).
            rng_seed: semente para reprodutibilidade.
            epsilon: pequeno valor numérico para evitar divisões por zero.
        """
        # Dados do problema
        self.cidades = cidades
        self.num_cidades = len(cidades)

        # Hiperparâmetros com validação básica
        self.tamanho_populacao = max(self.MIN_POPULACAO, int(tamanho_populacao))
        self.num_geracoes = max(self.MIN_GERACOES, int(num_geracoes))
        self.taxa_mutacao = float(max(0.0, min(1.0, taxa_mutacao)))
        self.taxa_elitismo = float(max(0.0, min(self.MAX_ELITISMO, taxa_elitismo)))

        # Métodos configuráveis
        self.metodo_selecao = metodo_selecao
        self.metodo_crossover = metodo_crossover
        self.metodo_mutacao = metodo_mutacao
        self.torneio_k = max(self.MIN_TORNEIO_SIZE, int(torneio_k))
        self.epsilon = epsilon

        # Se a matriz de distâncias não for passada, pré-computamos (ganho de performance)
        # PARA APRESENTAÇÃO: "Pré-calculamos a matriz para acelerar as avaliações."
        if dist_matrix is None:
            self.dist_matrix = utils.calcular_matriz_distancias(self.cidades)
        else:
            self.dist_matrix = dist_matrix

        # Inicializamos estrutura de população e históricos
        self.populacao: List[Individuo] = []
        self.historico_melhores: List[float] = []
        # historico_medias será criado durante a primeira geração

        # Opcional: fixar semente para tornar execuções reproduzíveis
        if rng_seed is not None:
            random.seed(rng_seed)

    # -----------------------
    # Criação / Inicialização
    # -----------------------
    def _criar_populacao_inicial(self):
        """Gera a população inicial com rotas aleatórias."""
        self.populacao = []
        for _ in range(self.tamanho_populacao):
            rota = utils.criar_rota_aleatoria(self.num_cidades)
            # PARA APRESENTAÇÃO: "Cada indivíduo calcula sua distância ao ser criado."
            self.populacao.append(Individuo(rota, self.cidades, self.dist_matrix, self.epsilon))

    # -----------------------
    # Seleções
    # -----------------------
    def _selecao_por_torneio(self) -> Individuo:
        """Escolhe K indivíduos aleatórios e retorna o melhor entre eles."""
        # Use random.sample para evitar repetição no torneio
        participantes = random.sample(self.populacao, k=min(self.torneio_k, len(self.populacao)))
        # Retorna o indivíduo com maior fitness (fitness = 1/distância)
        return max(participantes, key=lambda ind: ind.fitness)

    def _selecao_por_roleta(self) -> Individuo:
        """Seleção proporcional ao fitness (roleta)."""
        total = sum(ind.fitness for ind in self.populacao)
        # Caso degenerate (soma zero), fallback para torneio
        if total <= 0:
            return self._selecao_por_torneio()
        pick = random.uniform(0, total)
        current = 0.0
        for ind in self.populacao:
            current += ind.fitness
            if current >= pick:
                return ind
        # Retorno seguro (último indivíduo), raramente usado
        return self.populacao[-1]

    def _selecao_por_ranking(self) -> Individuo:
        """Seleção por ranking: pesos lineares pela posição."""
        # Ordena pela distância (menor é melhor)
        pop_sorted = sorted(self.populacao, key=lambda ind: ind.distancia)
        n = len(pop_sorted)
        # Peso linear: melhor posição tem peso n, pior tem peso 1
        pesos = [n - i for i in range(n)]
        escolha = random.choices(pop_sorted, weights=pesos, k=1)[0]
        return escolha

    def _selecionar_pai(self) -> Individuo:
        """Roteador para escolher método de seleção configurado."""
        metodos_disponiveis = {
            "torneio": self._selecao_por_torneio,
            "roleta": self._selecao_por_roleta,
            "ranking": self._selecao_por_ranking,
        }
        metodo = metodos_disponiveis.get(self.metodo_selecao, self._selecao_por_torneio)
        return metodo()

    def _selecionar_pais_distintos(self):
        """Tenta selecionar dois pais diferentes, evitando cópias idênticas."""
        pai1 = self._selecionar_pai()
        pai2 = self._selecionar_pai()
        tentativas = 0
        # Evita que pai1 == pai2 em excesso (ajuda diversidade)
        while pai1.rota == pai2.rota and tentativas < self.MAX_SELECTION_ATTEMPTS:
            pai2 = self._selecionar_pai()
            tentativas += 1
        return pai1, pai2

    # -----------------------
    # Crossovers
    # -----------------------
    def _crossover_ox(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        """Ordered Crossover (OX)."""
        # Escolhe dois cortes e copia o segmento do pai1
        a, b = sorted(random.sample(range(self.num_cidades), 2))
        filho = [None] * self.num_cidades
        filho[a:b + 1] = pai1.rota[a:b + 1]

        # Posição de inserção no filho e iteração no pai2 (circular)
        pos_filho = (b + 1) % self.num_cidades
        pos_pai2 = (b + 1) % self.num_cidades

        preenchidos = set(pai1.rota[a:b + 1])
        # Segurança: limite de iterações para evitar loops infinitos
        max_iteracoes = self.num_cidades * 2
        iteracoes = 0

        # Preenche o resto do filho seguindo a ordem do pai2, pulando genes já copiados
        while None in filho and iteracoes < max_iteracoes:
            gene = pai2.rota[pos_pai2]
            if gene not in preenchidos:
                # Avança até encontrar posição livre
                while filho[pos_filho] is not None:
                    pos_filho = (pos_filho + 1) % self.num_cidades
                filho[pos_filho] = gene
                preenchidos.add(gene)
                pos_filho = (pos_filho + 1) % self.num_cidades
            pos_pai2 = (pos_pai2 + 1) % self.num_cidades
            iteracoes += 1

        # Caso raríssimo: ainda existem posições None, preenche com os faltantes
        if None in filho:
            faltantes = [g for g in range(self.num_cidades) if g not in preenchidos]
            for i, val in enumerate(filho):
                if val is None and faltantes:
                    filho[i] = faltantes.pop(0)

        # PARA APRESENTAÇÃO: "OX preserva a ordem relativa dos genes, importante em permutações."
        return filho

    def _crossover_pmx(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        """Partially Mapped Crossover (PMX)."""
        # PMX mapeia um segmento do pai2 para o filho e resolve conflitos com mapeamento
        a, b = sorted(random.sample(range(self.num_cidades), 2))
        filho = [None] * self.num_cidades
        filho[a:b + 1] = pai2.rota[a:b + 1]

        # Cria mapeamento dos genes que foram trocados
        mapeamento = {}
        for i in range(a, b + 1):
            mapeamento[pai2.rota[i]] = pai1.rota[i]

        # Para posições fora do segmento, tenta manter genes do pai1 ajustando via mapeamento
        for i in range(self.num_cidades):
            if a <= i <= b:
                continue
            gene = pai1.rota[i]
            tentativas = 0
            while gene in filho and tentativas < self.num_cidades:
                gene = mapeamento.get(gene, gene)
                tentativas += 1
            filho[i] = gene

        # Se ainda houver posições None, preenche com chromosomes faltantes
        if None in filho:
            usados = set(g for g in filho if g is not None)
            faltantes = [g for g in range(self.num_cidades) if g not in usados]
            idx = 0
            for i in range(self.num_cidades):
                if filho[i] is None:
                    filho[i] = faltantes[idx]
                    idx += 1

        # PARA APRESENTAÇÃO: "PMX é mais complexo que OX, mas preserva mapeamentos parciais entre pais."
        return filho

    def _aplicar_crossover(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        """Chama o método de crossover configurado."""
        if self.metodo_crossover == "pmx":
            return self._crossover_pmx(pai1, pai2)
        else:
            return self._crossover_ox(pai1, pai2)

    # -----------------------
    # Mutações
    # -----------------------
    def _mutacao_swap(self, rota: List[int]) -> List[int]:
        """Mutação por swap: troca duas cidades."""
        rota_mut = rota[:]
        # Número de swaps determinado pela taxa_mutacao * tamanho
        max_swaps = max(1, int(self.taxa_mutacao * self.num_cidades))
        for _ in range(max_swaps):
            # Aqui checamos aleatoriamente se aplicamos o swap (probabilístico)
            if random.random() < self.taxa_mutacao:
                i, j = random.sample(range(self.num_cidades), 2)
                rota_mut[i], rota_mut[j] = rota_mut[j], rota_mut[i]
                # Retornamos após o primeiro swap efetivo para manter simplicidade
                return rota_mut
        return rota_mut

    def _mutacao_inversion(self, rota: List[int]) -> List[int]:
        """Mutação por inversão: inverte um segmento da rota."""
        rota_mut = rota[:]
        if random.random() < self.taxa_mutacao:
            i, j = sorted(random.sample(range(self.num_cidades), 2))
            # Inverte segmento i..j (inclusivo)
            rota_mut[i:j + 1] = list(reversed(rota_mut[i:j + 1]))
        return rota_mut

    def _aplicar_mutacao(self, rota: List[int]) -> List[int]:
        """Aplica o operador de mutação configurado."""
        if self.metodo_mutacao == "inversion":
            return self._mutacao_inversion(rota)
        else:
            return self._mutacao_swap(rota)

    # -----------------------
    # Operações utilitárias
    # -----------------------
    def _get_melhor_individuo(self) -> Individuo:
        """Retorna o indivíduo com menor distância da população atual."""
        return min(self.populacao, key=lambda ind: ind.distancia)

    def _aplicar_elitismo(self) -> List[Individuo]:
        """Retorna clones dos 'num_elite' melhores indivíduos."""
        num_elite = int(self.tamanho_populacao * self.taxa_elitismo)
        if num_elite == 0:
            return []
        # Ordena por distância (menor distância = melhor)
        elites = sorted(self.populacao, key=lambda ind: ind.distancia)[:num_elite]
        # Retorna cópias (para não compartilhar referências)
        return [e.copy() for e in elites]

    def _criar_individuo_aleatorio(self) -> Individuo:
        """Cria um indivíduo com rota aleatória."""
        rota = utils.criar_rota_aleatoria(self.num_cidades)
        return Individuo(rota, self.cidades, self.dist_matrix, self.epsilon)

    # -----------------------
    # Execução principal
    # -----------------------
    def executar(self) -> Individuo:
        """
        Executa o Algoritmo Genético e retorna o melhor indivíduo encontrado.

        Retorna:
            Individuo: melhor solução encontrada durante a execução.
        """
        # Mensagem inicial para o console
        print(f"\n[AG] Iniciando | Sel={self.metodo_selecao} | Cross={self.metodo_crossover} | Mut={self.metodo_mutacao}")

        # 1) inicializa população
        self._criar_populacao_inicial()
        melhor_global = self._get_melhor_individuo().copy()

        # inicializa históricos: guardamos o melhor e a média da população na geração 0
        self.historico_melhores = [melhor_global.distancia]
        self.historico_medias = [sum(ind.distancia for ind in self.populacao) / len(self.populacao)]

        # loop principal de gerações
        # PARA APRESENTAÇÃO: "Aqui será executada a evolução geracional por num_geracoes steps."
        for ger in range(1, self.num_geracoes + 1):

            # Elitismo: preserva melhores indivíduos para a próxima geração
            nova_pop: List[Individuo] = self._aplicar_elitismo()

            # Preenche o restante da população com filhos gerados por crossover + mutação
            attempts = 0
            while len(nova_pop) < self.tamanho_populacao:
                pai1, pai2 = self._selecionar_pais_distintos()

                # Geração do filho (crossover seguido de mutação)
                filho_rota = self._aplicar_crossover(pai1, pai2)
                filho_rota = self._aplicar_mutacao(filho_rota)

                # Cria novo indivíduo e adiciona à nova população
                novo_ind = Individuo(filho_rota, self.cidades, self.dist_matrix, self.epsilon)
                nova_pop.append(novo_ind)

                attempts += 1
                # Segurança: se tentativas estourarem, preenche com aleatórios para evitar deadlock
                if attempts > self.tamanho_populacao * self.MAX_POPULATION_FILL_ATTEMPTS_MULTIPLIER:
                    while len(nova_pop) < self.tamanho_populacao:
                        nova_pop.append(self._criar_individuo_aleatorio())
                    break

            # Substitui a população atual pela nova
            self.populacao = nova_pop

            # Avaliação da geração
            melhor_da_geracao = self._get_melhor_individuo()
            self.historico_melhores.append(melhor_da_geracao.distancia)

            # Coleta explícita da MÉDIA DA POPULAÇÃO (requisito do projeto)
            media_dist = sum(ind.distancia for ind in self.populacao) / len(self.populacao)
            self.historico_medias.append(media_dist)

            # Atualiza o melhor global se encontramos solução melhor nesta geração
            if melhor_da_geracao.distancia < melhor_global.distancia:
                melhor_global = melhor_da_geracao.copy()
                # PARA APRESENTAÇÃO: "Atualizamos o melhor global quando encontramos uma rota menor."

        # Final do AG: imprimimos e retornamos o melhor encontrado
        print(f"[AG] Concluído. Melhor Distância: {melhor_global.distancia:.4f}")
        # PARA APRESENTAÇÃO: "O método executar() devolve o melhor indivíduo; os históricos ficarão disponíveis para plotagem."
        return melhor_global
