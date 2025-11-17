# Dentro de algoritmo_genetico.py

import random
import utils

# --- CLASSE 1: O INDIVÍDUO (REPRESENTAÇÃO) ---

class Individuo:
    """
    Representa uma única solução (um "cromossomo") para o PCV.
    Guarda a rota e seu custo (fitness).
    """
    def __init__(self, rota, cidades):
        self.rota = rota
        self.cidades = cidades
        
        # Calcula e armazena a distância e o fitness
        self.distancia = utils.calcular_distancia_total(self.rota, self.cidades)
        
        # Fitness é o inverso da distância. Queremos MAXIMIZAR o fitness.
        # Distância menor = Fitness MAIOR.
        self.fitness = 1 / self.distancia

    def __repr__(self):
        # Uma forma fácil de imprimir o indivíduo e ver seu custo
        return f"[Indivíduo: Dist={self.distancia:.2f} | Rota={self.rota[:4]}...]"


# --- CLASSE 2: O ALGORITMO (GERENCIA A POPULAÇÃO) ---

class AlgoritmoGenetico:
    """
    Gerencia todo o processo de evolução da população
    de indivíduos (soluções).
    """
    def __init__(self, cidades, tamanho_populacao, num_geracoes, taxa_mutacao):
        # 1. Dados do Problema
        self.cidades = cidades
        self.num_cidades = len(cidades)
        
        # 2. Hiperparâmetros do AG
        self.tamanho_populacao = tamanho_populacao
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        
        # 3. Armazena a população
        self.populacao = [] # Lista de objetos 'Individuo'
        
    # Dentro da classe 'AlgoritmoGenetico' (adicione estes métodos):

    def _selecao_por_torneio(self, K=3):
        """
        Seleciona um indivíduo para reprodução usando Seleção por Torneio.
        K = tamanho do torneio (quantos indivíduos "competem").
        """
        # 1. Pega K indivíduos aleatórios da população (sem repetição)
        participantes = random.sample(self.populacao, K)
        
        # 2. Encontra o "vencedor" (o que tiver o MAIOR fitness)
        #    Lembre-se: maior fitness = menor distância
        vencedor = max(participantes, key=lambda ind: ind.fitness)
        
        return vencedor

    def _crossover_ordenado(self, pai1, pai2):
        """
        Executa o Ordered Crossover (OX) para criar uma nova rota (filho).
        Este crossover é específico para problemas de permutação como o PCV.
        """
        # [Image of Ordered Crossover (OX) for TSP]
        
        # 1. Pega as rotas (listas) dos objetos Indivíduo
        rota_pai1 = pai1.rota
        rota_pai2 = pai2.rota
        
        # 2. Escolhe dois pontos de corte aleatórios
        #    sorted() garante que 'inicio' seja sempre menor que 'fim'
        inicio, fim = sorted(random.sample(range(self.num_cidades), 2))
        
        # 3. Cria o "miolo" do filho com a fatia do pai1
        fatia_pai1 = rota_pai1[inicio : fim + 1]
        
        # 4. Inicializa o filho com "None"
        rota_filho = [None] * self.num_cidades
        
        # 5. Copia a fatia do pai1 para o filho
        rota_filho[inicio : fim + 1] = fatia_pai1
        
        # 6. Preenche o restante do filho com os genes do pai2
        
        # Ponteiro para a posição atual no pai2
        ponteiro_pai2 = 0 
        # Ponteiro para a posição atual no filho
        ponteiro_filho = 0 
        
        while None in rota_filho:
            # Encontra a posição de preenchimento no filho (pula o miolo)
            if rota_filho[ponteiro_filho] is not None:
                ponteiro_filho = (ponteiro_filho + 1) % self.num_cidades
                continue # Pula para a próxima posição do filho
            
            # Pega a cidade do pai2
            cidade_pai2 = rota_pai2[ponteiro_pai2]
            ponteiro_pai2 = (ponteiro_pai2 + 1) % self.num_cidades
            
            # Se a cidade do pai2 AINDA NÃO ESTÁ no filho, adiciona
            if cidade_pai2 not in fatia_pai1:
                rota_filho[ponteiro_filho] = cidade_pai2
        
        return rota_filho

    def _criar_populacao_inicial(self):
        """
        Gera a primeira população de soluções aleatórias.
        """
        print("Criando população inicial...")
        for _ in range(self.tamanho_populacao):
            # 1. Gera uma rota aleatória
            rota_aleatoria = utils.criar_rota_aleatoria(self.num_cidades)
            
            # 2. Cria um objeto 'Individuo' com essa rota
            novo_individuo = Individuo(rota=rota_aleatoria, cidades=self.cidades)
            
            # 3. Adiciona o novo indivíduo à população
            self.populacao.append(novo_individuo)
        
        print(f"População inicial criada com {len(self.populacao)} indivíduos.")
        
    def _mutacao_por_troca(self, rota):
        """
        Aplica a mutação por Troca (Swap) em uma rota.
        Troca duas cidades de posição aleatoriamente.
        """
        # Copia a rota para não alterar o original (importante!)
        rota_mutada = rota[:] 
        
        # Sorteia se a mutação VAI ocorrer
        if random.random() < self.taxa_mutacao:
            # Sorteia duas posições (índices) para trocar
            idx1, idx2 = random.sample(range(self.num_cidades), 2)
            
            # Realiza a troca (swap)
            # Sintaxe elegante do Python para trocar valores
            rota_mutada[idx1], rota_mutada[idx2] = rota_mutada[idx2], rota_mutada[idx1]
            
        return rota_mutada

    def _get_melhor_individuo(self):
        """Helper para encontrar o melhor indivíduo na população atual."""
        # 'min' na 'distancia' é o mesmo que 'max' no 'fitness'
        return min(self.populacao, key=lambda ind: ind.distancia)

    def executar(self):
        """
        Executa o loop principal do Algoritmo Genético por N gerações.
        """
        # 1. Cria a população inicial
        self._criar_populacao_inicial()
        
        # 2. Guarda o melhor indivíduo da Geração 0 (aleatória)
        melhor_global = self._get_melhor_individuo()
        print(f"Melhor distância inicial (Geração 0): {melhor_global.distancia:.2f}")
        
        # 3. Executa o loop de gerações (Critério de Parada)
        for i in range(1, self.num_geracoes + 1):
            nova_populacao = []
            
            # --- Elitismo ---
            # Guarda o melhor indivíduo da geração anterior
            # (Conforme Análise 4)
            melhor_da_geracao_anterior = self._get_melhor_individuo()
            nova_populacao.append(melhor_da_geracao_anterior)
            
            # Preenche o resto da população com filhos (N-1)
            while len(nova_populacao) < self.tamanho_populacao:
                # a. Seleção
                pai1 = self._selecao_por_torneio()
                pai2 = self._selecao_por_torneio()
                
                # b. Crossover
                rota_filho = self._crossover_ordenado(pai1, pai2)
                
                # c. Mutação
                rota_filho_mutada = self._mutacao_por_troca(rota_filho)
                
                # d. Cria o novo indivíduo (calcula fitness)
                novo_filho = Individuo(rota=rota_filho_mutada, cidades=self.cidades)
                nova_populacao.append(novo_filho)
            
            # Atualiza a população
            self.populacao = nova_populacao
            
            # Acompanha o melhor global
            melhor_atual = self._get_melhor_individuo()
            if melhor_atual.distancia < melhor_global.distancia:
                melhor_global = melhor_atual
            
            # Print de log (para Gráfico de Convergência )
            if i % 10 == 0 or i == self.num_geracoes:
                print(f"Geração {i:3}: Melhor Distância = {melhor_atual.distancia:.2f} (Melhor Global: {melhor_global.distancia:.2f})")
        
        print("Evolução concluída.")
        return melhor_global