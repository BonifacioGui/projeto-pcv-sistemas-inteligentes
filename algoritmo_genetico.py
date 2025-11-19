# algoritmo_genetico.py
from typing import List, Tuple, Optional
import random
import utils  # DEPENDÊNCIA: Certifique-se de que seu arquivo utils.py tem as funções chamadas abaixo

class Individuo:
    """
    Representa uma única solução (uma rota).
    Calcula a distância assim que é criado para evitar processamento repetido.
    """
    # Constante para evitar divisão por zero se a distância for 0 (o que seria impossível no TSP, mas é segurança)
    DEFAULT_EPSILON = 1e-9

    def __init__(self, rota: List[int], cidades: List[Tuple[float, float]],
                 dist_matrix: Optional[List[List[float]]] = None,
                 epsilon: float = DEFAULT_EPSILON):
        self.rota: List[int] = rota[:]  # Copia a lista para evitar referência cruzada
        self.cidades = cidades
        self.dist_matrix = dist_matrix
        self.epsilon = epsilon

        # Chama o utils para calcular a distância total desta rota
        self.distancia: float = utils.calcular_distancia_total(self.rota, self.cidades, self.dist_matrix)
        
        # O Fitness é o inverso da distância (quanto menor a distância, maior o fitness)
        self.fitness: float = 1.0 / (self.distancia + self.epsilon)

    def copy(self) -> "Individuo":
        """Cria um clone deste indivíduo. Essencial para o Elitismo não alterar o original."""
        return Individuo(self.rota[:], self.cidades, self.dist_matrix, self.epsilon)

    def __repr__(self):
        return f"[Indivíduo: Dist={self.distancia:.4f} | Rota={self.rota[:6]}...]"

class AlgoritmoGenetico:
    """
    Classe principal que gerencia a evolução da população.
    """
    # Configurações de segurança (evitam loops infinitos e números mágicos)
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
        self.cidades = cidades
        self.num_cidades = len(cidades)
        
        # Validações para garantir que o usuário não passe números negativos ou inválidos
        self.tamanho_populacao = max(self.MIN_POPULACAO, int(tamanho_populacao))
        self.num_geracoes = max(self.MIN_GERACOES, int(num_geracoes))
        self.taxa_mutacao = float(max(0.0, min(1.0, taxa_mutacao)))
        self.taxa_elitismo = float(max(0.0, min(self.MAX_ELITISMO, taxa_elitismo)))
        
        self.metodo_selecao = metodo_selecao
        self.metodo_crossover = metodo_crossover
        self.metodo_mutacao = metodo_mutacao
        self.torneio_k = max(self.MIN_TORNEIO_SIZE, int(torneio_k))
        self.epsilon = epsilon

        # Se a matriz de distâncias não foi passada, calcula ela agora (ganho de performance)
        if dist_matrix is None:
            self.dist_matrix = utils.calcular_matriz_distancias(self.cidades)
        else:
            self.dist_matrix = dist_matrix

        # Listas para armazenar o estado atual
        self.populacao: List[Individuo] = []
        self.historico_melhores: List[float] = []

        # Seed para reprodutibilidade (se quiser que o teste dê sempre o mesmo resultado)
        if rng_seed is not None:
            random.seed(rng_seed)

    # -----------------------
    # Criação / Inicialização
    # -----------------------
    def _criar_populacao_inicial(self):
        """Gera a primeira geração com rotas totalmente aleatórias."""
        self.populacao = []
        for _ in range(self.tamanho_populacao):
            rota = utils.criar_rota_aleatoria(self.num_cidades)
            self.populacao.append(
                Individuo(rota, self.cidades, self.dist_matrix, self.epsilon)
            )

    # -----------------------
    # Métodos de Seleção
    # -----------------------
    def _selecao_por_torneio(self) -> Individuo:
        """Pega K indivíduos aleatórios e retorna o melhor deles."""
        participantes = random.sample(
            self.populacao, k=min(self.torneio_k, len(self.populacao))
        )
        return max(participantes, key=lambda ind: ind.fitness)

    def _selecao_por_roleta(self) -> Individuo:
        """Gira uma roleta viciada onde quem tem maior fitness tem maior chance."""
        total = sum(ind.fitness for ind in self.populacao)
        if total <= 0:
            return self._selecao_por_torneio() # Fallback
        
        pick = random.uniform(0, total)
        current = 0.0
        for ind in self.populacao:
            current += ind.fitness
            if current >= pick:
                return ind
        return self.populacao[-1]

    def _selecao_por_ranking(self) -> Individuo:
        """Ignora o valor numérico do fitness e usa apenas a posição (1º, 2º, 3º...)."""
        pop_sorted = sorted(self.populacao, key=lambda ind: ind.distancia)
        n = len(pop_sorted)
        pesos = [n - i for i in range(n)] # O primeiro ganha peso N, o último peso 1
        escolha = random.choices(pop_sorted, weights=pesos, k=1)[0]
        return escolha

    def _selecionar_pai(self) -> Individuo:
        """Roteador que escolhe qual função de seleção usar baseado na configuração."""
        metodos_disponiveis = {
            "torneio": self._selecao_por_torneio,
            "roleta": self._selecao_por_roleta,
            "ranking": self._selecao_por_ranking,
        }
        # Se o método não existir, usa torneio como padrão
        metodo = metodos_disponiveis.get(
            self.metodo_selecao, 
            self._selecao_por_torneio
        )
        return metodo()

    def _selecionar_pais_distintos(self) -> Tuple[Individuo, Individuo]:
        """Tenta selecionar dois pais que não sejam idênticos para garantir diversidade."""
        pai1 = self._selecionar_pai()
        pai2 = self._selecionar_pai()
        
        tentativas = 0
        # Tenta achar um par diferente até atingir o limite de tentativas
        while pai1.rota == pai2.rota and tentativas < self.MAX_SELECTION_ATTEMPTS:
            pai2 = self._selecionar_pai()
            tentativas += 1
            
        return pai1, pai2

    # -----------------------
    # Métodos de Crossover (Cruzamento)
    # -----------------------
    def _crossover_ox(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        """Ordered Crossover: Mantém a ordem relativa dos genes."""
        a, b = sorted(random.sample(range(self.num_cidades), 2))
        filho = [None] * self.num_cidades
        
        # 1. Copia um trecho do Pai 1
        filho[a:b + 1] = pai1.rota[a:b + 1]
        
        # 2. Preenche o resto com genes do Pai 2, na ordem que aparecem
        pos_filho = (b + 1) % self.num_cidades
        pos_pai2 = (b + 1) % self.num_cidades
        preenchidos = set(pai1.rota[a:b + 1])

        max_iteracoes = self.num_cidades * 2
        iteracoes = 0
        
        while None in filho and iteracoes < max_iteracoes:
            gene = pai2.rota[pos_pai2]
            if gene not in preenchidos:
                while filho[pos_filho] is not None:
                    pos_filho = (pos_filho + 1) % self.num_cidades
                filho[pos_filho] = gene
                preenchidos.add(gene)
                pos_filho = (pos_filho + 1) % self.num_cidades
            pos_pai2 = (pos_pai2 + 1) % self.num_cidades
            iteracoes += 1
            
        # Fallback de segurança (caso raro de falha lógica)
        if None in filho:
            faltantes = [g for g in range(self.num_cidades) if g not in preenchidos]
            for i, val in enumerate(filho):
                if val is None and faltantes:
                    filho[i] = faltantes.pop(0)
                    
        return filho

    def _crossover_pmx(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        """Partially Mapped Crossover: Usa mapeamento para resolver conflitos."""
        a, b = sorted(random.sample(range(self.num_cidades), 2))
        filho = [None] * self.num_cidades
        
        # Copia fatia do pai2
        filho[a:b + 1] = pai2.rota[a:b + 1]
        
        # Cria mapa de relacionamento
        mapeamento = {}
        for i in range(a, b + 1):
            mapeamento[pai2.rota[i]] = pai1.rota[i]
            
        # Preenche o resto resolvendo colisões
        for i in range(self.num_cidades):
            if a <= i <= b:
                continue
            gene = pai1.rota[i]
            tentativas = 0
            while gene in filho and tentativas < self.num_cidades:
                gene = mapeamento.get(gene, gene)
                tentativas += 1
            filho[i] = gene
            
        # Fallback de segurança
        if None in filho:
            usados = set(g for g in filho if g is not None)
            faltantes = [g for g in range(self.num_cidades) if g not in usados]
            idx = 0
            for i in range(self.num_cidades):
                if filho[i] is None:
                    filho[i] = faltantes[idx]
                    idx += 1
        return filho

    def _aplicar_crossover(self, pai1: Individuo, pai2: Individuo) -> List[int]:
        if self.metodo_crossover == "pmx":
            return self._crossover_pmx(pai1, pai2)
        else:
            return self._crossover_ox(pai1, pai2)

    # -----------------------
    # Métodos de Mutação
    # -----------------------
    def _mutacao_swap(self, rota: List[int]) -> List[int]:
        """Troca dois genes de lugar aleatoriamente."""
        rota_mut = rota[:]
        max_swaps = max(1, int(self.taxa_mutacao * self.num_cidades))
        
        for _ in range(max_swaps):
            if random.random() < self.taxa_mutacao:
                i, j = random.sample(range(self.num_cidades), 2)
                rota_mut[i], rota_mut[j] = rota_mut[j], rota_mut[i]
                return rota_mut

    def _mutacao_inversion(self, rota: List[int]) -> List[int]:
        """Inverte um pedaço da rota (bom para desenrolar caminhos cruzados)."""
        rota_mut = rota[:]
        if random.random() < self.taxa_mutacao:
            i, j = sorted(random.sample(range(self.num_cidades), 2))
            rota_mut[i:j + 1] = list(reversed(rota_mut[i:j + 1]))
        return rota_mut

    def _aplicar_mutacao(self, rota: List[int]) -> List[int]:
        if self.metodo_mutacao == "inversion":
            return self._mutacao_inversion(rota)
        else:
            return self._mutacao_swap(rota)

    # -----------------------
    # Execução Principal
    # -----------------------
    def _get_melhor_individuo(self) -> Individuo:
        return min(self.populacao, key=lambda ind: ind.distancia)

    def _aplicar_elitismo(self) -> List[Individuo]:
        """Salva os melhores da geração passada sem alterações."""
        num_elite = int(self.tamanho_populacao * self.taxa_elitismo)
        if num_elite == 0:
            return []
        
        elites = sorted(self.populacao, key=lambda ind: ind.distancia)[:num_elite]
        return [e.copy() for e in elites]

    def _criar_individuo_aleatorio(self) -> Individuo:
        rota = utils.criar_rota_aleatoria(self.num_cidades)
        return Individuo(rota, self.cidades, self.dist_matrix, self.epsilon)

    def executar(self) -> Individuo:
        print(f"\n[AG] Iniciando | Sel={self.metodo_selecao} | Cross={self.metodo_crossover} | Mut={self.metodo_mutacao}")
        
        self._criar_populacao_inicial()
        melhor_global = self._get_melhor_individuo()
        
        # Salva estado inicial
        self.historico_melhores = [melhor_global.distancia]
        media_inicial = sum(ind.distancia for ind in self.populacao) / len(self.populacao) # NOVO
        self.historico_medias = [media_inicial] # NOVO
        
        print(f"[AG] Melhor inicial: {melhor_global.distancia:.4f}")

        for ger in range(1, self.num_geracoes + 1):
            nova_pop: List[Individuo] = []
            
            # 1. Elitismo
            nova_pop.extend(self._aplicar_elitismo())
            
            # 2. Preenche população
            max_attempts = self.tamanho_populacao * self.MAX_POPULATION_FILL_ATTEMPTS_MULTIPLIER
            attempts = 0
            while len(nova_pop) < self.tamanho_populacao and attempts < max_attempts:
                attempts += 1
                pai1, pai2 = self._selecionar_pais_distintos()
                rota_filho = self._aplicar_crossover(pai1, pai2)
                rota_filho = self._aplicar_mutacao(rota_filho)
                novo = Individuo(rota_filho, self.cidades, self.dist_matrix, self.epsilon)
                nova_pop.append(novo)

            # 3. Fallback
            while len(nova_pop) < self.tamanho_populacao:
                nova_pop.append(self._criar_individuo_aleatorio())

            self.populacao = nova_pop
            
            # --- COLETA DE DADOS PARA O RELATÓRIO ---
            melhor_atual = self._get_melhor_individuo()
            if melhor_atual.distancia < melhor_global.distancia:
                melhor_global = melhor_atual
            
            # Calcula média da geração atual
            media_atual = sum(ind.distancia for ind in self.populacao) / len(self.populacao) # NOVO
            
            self.historico_melhores.append(melhor_atual.distancia)
            self.historico_medias.append(media_atual) # NOVO
            # ----------------------------------------
            
            if ger % 50 == 0 or ger == self.num_geracoes:
                print(f"[AG] Geração {ger:4} | Melhor={melhor_atual.distancia:.4f} | Média={media_atual:.4f}")

        return melhor_global