# algoritmo_genetico.py
import random
import copy
import utils

# --- CLASSE 1: O INDIVÍDUO (REPRESENTAÇÃO) ---

class Individuo:
    """
    Representa uma única solução (um "cromossomo") para o PCV.
    Guarda a rota, a lista de cidades (coordenadas), sua distância total e seu fitness.
    """
    def __init__(self, rota, cidades):
        """
        Construtor do Indivíduo.
        Calcula e armazena a distância e o fitness.
        """
        # 1. Armazena a rota (ex: [0, 4, 2, 1, 3])
        self.rota = rota
        # 2. Armazena a lista de coordenadas (para o cálculo)
        self.cidades = cidades

        # 3. Calcula a distância total da rota (chama utils.py)
        #    Espera-se que utils.calcular_distancia_total(retorne um float > 0)
        self.distancia = utils.calcular_distancia_total(self.rota, self.cidades)

        # 4. Converte o problema de minimização (distância)
        #    para maximização (fitness).
        #    Evita divisão por zero adicionando um epsilon mínimo.
        eps = 1e-9
        self.fitness = 1.0 / (self.distancia + eps)

    def __repr__(self):
        """
        Método especial para "representação" (como o objeto é impresso).
        Facilita a depuração.
        """
        # Mostra as primeiras 4 cidades da rota para não poluir a saída
        return f"[Indivíduo: Dist={self.distancia:.2f} | Rota={self.rota[:4]}...]"


# --- CLASSE 2: O ALGORITMO (GERENCIA A POPULAÇÃO) ---

class AlgoritmoGenetico:
    """
    Gerencia todo o processo de evolução da população de indivíduos (soluções).
    Parâmetros:
      - cidades: lista de coordenadas (ou estrutura aceita por utils)
      - tamanho_populacao: int
      - num_geracoes: int
      - taxa_mutacao: float [0..1]
      - taxa_elitismo: float [0..1]
      - metodo_selecao: 'torneio' | 'roleta'
      - metodo_mutacao: 'troca' | 'inversao'
      - metodo_crossover: 'ox' | 'pmx'
    """
    def __init__(self, cidades, tamanho_populacao, num_geracoes,
                 taxa_mutacao, taxa_elitismo=0.05,
                 metodo_selecao='torneio', metodo_mutacao='troca',
                 metodo_crossover='ox'):

        # 1. Dados do Problema
        self.cidades = cidades
        self.num_cidades = len(cidades)

        # 2. Hiperparâmetros do AG
        self.tamanho_populacao = tamanho_populacao
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_elitismo = taxa_elitismo

        # 3. Configuração dos Operadores
        self.metodo_selecao = metodo_selecao
        self.metodo_mutacao = metodo_mutacao
        self.metodo_crossover = metodo_crossover

        # 4. Armazena a população e o histórico
        self.populacao = []
        self.historico_melhores = []

    # ----------------------------
    # MÉTODOS DE SELEÇÃO
    # ----------------------------
    def _selecao_por_torneio(self, K=3):
        """
        Seleção por torneio: escolhe K indivíduos aleatórios e retorna o melhor.
        """
        K = max(2, min(K, len(self.populacao)))  # garante K válido
        participantes = random.sample(self.populacao, K)
        vencedor = max(participantes, key=lambda ind: ind.fitness)
        return vencedor

    def _selecao_por_roleta(self):
        """
        Seleção por roleta (Fitness Proportionate Selection).
        Se o fitness_total for ~0, escolhe aleatoriamente (fallback seguro).
        """
        fitness_total = sum(ind.fitness for ind in self.populacao)
        if fitness_total <= 0:
            # fallback: todos têm fitness 0 -> retorna aleatório
            return random.choice(self.populacao)

        ponto_sorteado = random.uniform(0, fitness_total)
        soma_acumulada = 0.0
        for individuo in self.populacao:
            soma_acumulada += individuo.fitness
            if soma_acumulada >= ponto_sorteado:
                return individuo

        # fallback por segurança (ponto flutuante)
        return self.populacao[-1]

    # ----------------------------
    # MÉTODOS DE MUTAÇÃO
    # ----------------------------
    def _mutacao_por_troca(self, rota):
        """
        Mutação por troca (swap): troca duas cidades de posição.
        """
        rota_mutada = rota[:]
        if random.random() < self.taxa_mutacao:
            idx1, idx2 = random.sample(range(self.num_cidades), 2)
            rota_mutada[idx1], rota_mutada[idx2] = rota_mutada[idx2], rota_mutada[idx1]
        return rota_mutada

    def _mutacao_por_inversao(self, rota):
        """
        Mutação por inversão (2-opt): inverte um segmento aleatório.
        """
        rota_mutada = rota[:]
        if random.random() < self.taxa_mutacao:
            inicio, fim = sorted(random.sample(range(self.num_cidades), 2))
            rota_mutada[inicio:fim+1] = list(reversed(rota_mutada[inicio:fim+1]))
        return rota_mutada

    # ----------------------------
    # MÉTODOS DE CROSSOVER
    # ----------------------------
    def _crossover_ordenado(self, pai1, pai2):
        """
        Ordered Crossover (OX).
        Gera um filho mantendo a subsequência do pai1 e preenchendo com genes do pai2
        na ordem em que aparecem, pulando duplicatas.
        Implementação robusta para evitar loops.
        """
        rota_pai1 = pai1.rota
        rota_pai2 = pai2.rota

        inicio, fim = sorted(random.sample(range(self.num_cidades), 2))

        # miolo copiado do pai1
        fatia_pai1 = rota_pai1[inicio:fim+1]
        rota_filho = [None] * self.num_cidades
        rota_filho[inicio:fim+1] = fatia_pai1

        ponteiro_pai2 = 0
        ponteiro_filho = 0

        # Preenche todas as posições None
        while None in rota_filho:
            # Avança ponteiro_filho até encontrar posição livre
            while rota_filho[ponteiro_filho] is not None:
                ponteiro_filho = (ponteiro_filho + 1) % self.num_cidades

            # Procura próxima cidade válida do pai2
            cidade_pai2 = rota_pai2[ponteiro_pai2]
            ponteiro_pai2 = (ponteiro_pai2 + 1) % self.num_cidades

            # Se cidade não está no miolo copiado do pai1, adiciona
            if cidade_pai2 not in fatia_pai1:
                rota_filho[ponteiro_filho] = cidade_pai2
                # Após preencher, avança ponteiro_filho para próxima posição livre
                ponteiro_filho = (ponteiro_filho + 1) % self.num_cidades
            # se estava no miolo, apenas continua o loop (ponteiro_filho não avança aqui
            # pois não foi preenchido; ponteiro_pai2 já avançou)

        return rota_filho

    def _crossover_pmx(self, pai1, pai2):
        """
        Partially Mapped Crossover (PMX).
        Implementação que cria um mapeamento e resolve cadeias de mapeamento corretamente.
        """
        rota_pai1 = pai1.rota
        rota_pai2 = pai2.rota

        inicio, fim = sorted(random.sample(range(self.num_cidades), 2))

        # Filho começa como cópia do pai1
        rota_filho = rota_pai1[:]

        # Fatia do pai2 que será inserida
        fatia_pai2 = rota_pai2[inicio:fim+1]
        fatia_pai1 = rota_pai1[inicio:fim+1]

        # Mapeamento p2 -> p1 (usado para substituir duplicatas)
        # Ex: se fatia_pai1=[A,B,C] e fatia_pai2=[D,E,F], entao map_p2p1 = {D:A, E:B, F:C}
        map_p2p1 = {fatia_pai2[i]: fatia_pai1[i] for i in range(len(fatia_pai2))}
        # Também cria o mapeamento inverso p1 -> p2 (útil se precisar resolver em outra direção)
        map_p1p2 = {fatia_pai1[i]: fatia_pai2[i] for i in range(len(fatia_pai1))}

        # Copia a fatia do pai2 para o filho
        rota_filho[inicio:fim+1] = fatia_pai2

        # Corrige duplicatas fora da fatia
        for i in range(self.num_cidades):
            if inicio <= i <= fim:
                continue  # posição dentro do miolo já está correta

            valor = rota_filho[i]
            # enquanto o valor estiver presente na fatia copiada (causando duplicata)
            # substitui usando o mapeamento p2->p1 até não ser uma cidade da fatia_pai2
            # Observação: valor sempre estará em map_p2p1 apenas se for da fatia_pai2
            while valor in map_p2p1:
                valor = map_p2p1[valor]
            rota_filho[i] = valor

        return rota_filho

    # ----------------------------
    # POPULAÇÃO INICIAL E UTILITÁRIOS
    # ----------------------------
    def _criar_populacao_inicial(self):
        """
        Gera a primeira população de soluções aleatórias.
        """
        self.populacao = []
        for _ in range(self.tamanho_populacao):
            rota_aleatoria = utils.criar_rota_aleatoria(self.num_cidades)
            novo_individuo = Individuo(rota=rota_aleatoria, cidades=self.cidades)
            self.populacao.append(novo_individuo)

    def _get_melhor_individuo(self):
        """
        Retorna o indivíduo com menor distância (melhor solução).
        """
        return min(self.populacao, key=lambda ind: ind.distancia)

    # ----------------------------
    # LOOP PRINCIPAL
    # ----------------------------
    def executar(self):
        """
        Executa o loop principal do Algoritmo Genético por num_geracoes.
        Retorna o melhor indivíduo encontrado (cópia profunda).
        """
        # Validações simples de métodos escolhidos
        if self.metodo_selecao not in ('torneio', 'roleta'):
            raise ValueError(f"método_selecao inválido: {self.metodo_selecao}")
        if self.metodo_mutacao not in ('troca', 'inversao'):
            raise ValueError(f"método_mutacao inválido: {self.metodo_mutacao}")
        if self.metodo_crossover not in ('ox', 'pmx'):
            raise ValueError(f"metodo_crossover inválido: {self.metodo_crossover}")

        print(f"Iniciando AG: Seleção='{self.metodo_selecao}', Mutação='{self.metodo_mutacao}', Crossover='{self.metodo_crossover}', Elitismo={self.taxa_elitismo*100:.1f}%")

        # 1. Cria a população inicial
        self._criar_populacao_inicial()

        # 2. Guarda o melhor indivíduo da Geração 0 (aleatória)
        melhor_global = copy.deepcopy(self._get_melhor_individuo())
        self.historico_melhores = [melhor_global.distancia]
        print(f"Melhor distância inicial (Geração 0): {melhor_global.distancia:.2f}")

        # 3. Loop de gerações
        for geracao in range(1, self.num_geracoes + 1):
            nova_populacao = []

            # --- Elitismo ---
            num_elite = int(self.tamanho_populacao * self.taxa_elitismo)
            if num_elite > 0:
                pop_ordenada = sorted(self.populacao, key=lambda ind: ind.distancia)
                # copia profunda dos elites para evitar links indesejados (opcional)
                elites = [copy.deepcopy(ind) for ind in pop_ordenada[:num_elite]]
                nova_populacao.extend(elites)

            # Preenche o resto da população com filhos
            while len(nova_populacao) < self.tamanho_populacao:
                # --- Seleção de pais ---
                if self.metodo_selecao == 'torneio':
                    pai1 = self._selecao_por_torneio()
                    pai2 = self._selecao_por_torneio()
                    # Evita pai1 == pai2 quando possível (aumenta diversidade)
                    tentativa = 0
                    while pai2 is pai1 and tentativa < 10:
                        pai2 = self._selecao_por_torneio()
                        tentativa += 1
                else:  # 'roleta'
                    pai1 = self._selecao_por_roleta()
                    pai2 = self._selecao_por_roleta()
                    tentativa = 0
                    while pai2 is pai1 and tentativa < 10:
                        pai2 = self._selecao_por_roleta()
                        tentativa += 1

                # --- Crossover ---
                if self.metodo_crossover == 'ox':
                    rota_filho = self._crossover_ordenado(pai1, pai2)
                else:  # 'pmx'
                    rota_filho = self._crossover_pmx(pai1, pai2)

                # --- Mutação ---
                if self.metodo_mutacao == 'troca':
                    rota_filho_mutada = self._mutacao_por_troca(rota_filho)
                else:  # 'inversao'
                    rota_filho_mutada = self._mutacao_por_inversao(rota_filho)

                # Cria o novo indivíduo (calcula distância/fintess)
                novo_filho = Individuo(rota=rota_filho_mutada, cidades=self.cidades)
                nova_populacao.append(novo_filho)

            # Atualiza a população
            self.populacao = nova_populacao

            # Atualiza melhor global (faz cópia para não manter referência)
            melhor_atual = self._get_melhor_individuo()
            if melhor_atual.distancia < melhor_global.distancia:
                melhor_global = copy.deepcopy(melhor_atual)

            # Salva histórico (melhor desta geração)
            self.historico_melhores.append(melhor_atual.distancia)

            # Log de progresso a cada 10 gerações e na final
            if geracao % 10 == 0 or geracao == self.num_geracoes:
                print(f"Geração {geracao:4}: Melhor Distância = {melhor_atual.distancia:.2f} (Melhor Global: {melhor_global.distancia:.2f})")

        print("Evolução concluída.")
        return melhor_global
