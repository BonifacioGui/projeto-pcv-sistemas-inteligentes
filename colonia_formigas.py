# Dentro de colonia_formigas.py

import random
import utils
import numpy as np # Vamos usar numpy para os cálculos matriciais

class ACO:
    """
    Implementa o algoritmo Ant Colony Optimization (ACO) para o PCV.
    """
    def __init__(self, cidades, num_formigas, num_iteracoes,
                 alfa=1.0, beta=5.0, rho=0.1, Q=100):
        
        # 1. Dados do Problema
        self.cidades = cidades
        self.num_cidades = len(cidades)
        
        # 2. Hiperparâmetros do ACO
        self.num_formigas = num_formigas # Tamanho da "população"
        self.num_iteracoes = num_iteracoes   # "Gerações"
        
        # Influência do feromônio
        self.alfa = alfa  
        # Influência da visibilidade (distância)
        self.beta = beta  
        # Taxa de evaporação do feromônio
        self.rho = rho    
        # Quantidade de feromônio a ser depositado
        self.Q = Q        
        
        # 3. Matrizes do ACO
        
        # Matriz de feromônio (inicia com 1 em todos os caminhos)
        self.feromonios = np.ones((self.num_cidades, self.num_cidades))
        
        # Matriz de Heurística/Visibilidade (calculada uma vez)
        self.heuristicas = self._calcular_matriz_heuristicas()

    def _calcular_matriz_heuristicas(self):
        """
        Calcula a matriz de visibilidade (1 / distancia).
        Caminhos mais curtos têm maior heurística.
        """
        heuristicas = np.zeros((self.num_cidades, self.num_cidades))
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                # Calcula a distância entre a cidade i e j
                dist = utils.calcular_distancia(self.cidades[i], self.cidades[j])
                
                # Heurística é o inverso da distância (evita divisão por zero)
                valor_heuristica = 1.0 / (dist + 1e-10) 
                
                heuristicas[i][j] = valor_heuristica
                heuristicas[j][i] = valor_heuristica
        
        return heuristicas

    # --- Próximos Passos (a implementar) ---

    # Dentro da classe ACO, SUBSTITUA os métodos vazios por estes:

    def executar(self):
        """
        Executa o loop principal do ACO por N iterações.
        """
        print(f"Iniciando ACO: Formigas={self.num_formigas}, Iterações={self.num_iteracoes}")
        
        # Guarda a melhor rota encontrada em toda a execução
        melhor_rota_global = None
        menor_distancia_global = float('inf') # Começa com infinito
        
        # Loop principal (Iterações)
        for i in range(1, self.num_iteracoes + 1):
            
            # 1. Construir soluções (uma para cada formiga)
            rotas_das_formigas = []
            for _ in range(self.num_formigas):
                rota = self._construir_solucao()
                rotas_das_formigas.append(rota)
            
            # 2. Atualizar o Feromônio (Evaporação + Depósito)
            self._atualizar_feromonio(rotas_das_formigas)
            
            # 3. Encontrar a melhor rota desta iteração
            # (e atualizar a melhor rota global)
            melhor_dist_iteracao = float('inf')
            
            for rota in rotas_das_formigas:
                dist_rota = utils.calcular_distancia_total(rota, self.cidades)
                
                if dist_rota < melhor_dist_iteracao:
                    melhor_dist_iteracao = dist_rota
                
                if dist_rota < menor_distancia_global:
                    menor_distancia_global = dist_rota
                    melhor_rota_global = rota
            
            # Print de log (para acompanhar a convergência)
            if i % 10 == 0 or i == self.num_iteracoes:
                print(f"Iteração {i:3}: Melhor Distância = {melhor_dist_iteracao:.2f} (Melhor Global: {menor_distancia_global:.2f})")

        print("Evolução do ACO concluída.")
        # Retorna a melhor rota (lista de índices) e sua distância
        return melhor_rota_global, menor_distancia_global

    def _construir_solucao(self):
        """
        Simula uma formiga construindo uma rota completa.
        A escolha da próxima cidade é baseada em probabilidade.
        """
        # Começa a rota em uma cidade aleatória
        cidade_inicial = random.randint(0, self.num_cidades - 1)
        rota = [cidade_inicial]
        
        # Cidades que ainda precisam ser visitadas
        cidades_a_visitar = set(range(self.num_cidades))
        cidades_a_visitar.remove(cidade_inicial)
        
        # Loop: Constrói o resto da rota (N-1 cidades)
        while cidades_a_visitar:
            cidade_atual = rota[-1] # A última cidade adicionada
            
            # Calcula a "desejabilidade" de ir para cada próxima cidade
            probabilidades = []
            cidades_possiveis = []
            
            for proxima_cidade in cidades_a_visitar:
                # Pega o valor do feromônio e da heurística
                feromonio = self.feromonios[cidade_atual][proxima_cidade]
                heuristica = self.heuristicas[cidade_atual][proxima_cidade]
                
                # Fórmula de Probabilidade do ACO
                # [Image of ACO probability formula for TSP]
                prob = (feromonio ** self.alfa) * (heuristica ** self.beta)
                
                probabilidades.append(prob)
                cidades_possiveis.append(proxima_cidade)
            
            # Normaliza as probabilidades (para que somem 1)
            soma_probs = sum(probabilidades)
            probabilidades_normalizadas = [p / (soma_probs + 1e-10) for p in probabilidades]
            
            # Escolhe a próxima cidade com base nas probabilidades (roleta)
            proxima_cidade_escolhida = random.choices(
                cidades_possiveis, 
                weights=probabilidades_normalizadas, 
                k=1
            )[0]
            
            # Adiciona a cidade escolhida à rota
            rota.append(proxima_cidade_escolhida)
            cidades_a_visitar.remove(proxima_cidade_escolhida)
            
        return rota

    def _atualizar_feromonio(self, rotas_das_formigas):
        """
        Atualiza a matriz de feromônio:
        1. Evapora o feromônio antigo em TODOS os caminhos.
        2. Deposita novo feromônio nos caminhos usados pelas formigas.
        """
        
        # 1. Evaporação
        # Multiplica todos os caminhos por (1 - rho)
        # Ex: se rho=0.1, 90% do feromônio permanece
        self.feromonios *= (1 - self.rho)
        
        # 2. Depósito
        for rota in rotas_das_formigas:
            # Calcula a distância (custo) da rota
            distancia_rota = utils.calcular_distancia_total(rota, self.cidades)
            
            # A quantidade de feromônio a depositar é inversamente
            # proporcional à distância (rotas mais curtas depositam mais)
            feromonio_a_depositar = self.Q / distancia_rota
            
            # Deposita o feromônio em cada segmento (i, j) da rota
            for i in range(self.num_cidades):
                cidade_i = rota[i]
                cidade_j = rota[(i + 1) % self.num_cidades] # Pega a próxima (ou a 1ª se for a última)
                
                self.feromonios[cidade_i][cidade_j] += feromonio_a_depositar
                self.feromonios[cidade_j][cidade_i] += feromonio_a_depositar # Caminho simétrico