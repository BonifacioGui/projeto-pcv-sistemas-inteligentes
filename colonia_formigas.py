import random
import numpy as np
import utils

class ACO:
    """
    Implementa o algoritmo Ant Colony Optimization (ACO) para o PCV.
    """

    def __init__(self, cidades, num_formigas, num_iteracoes, alfa, beta, rho, Q):
        self.cidades = cidades
        self.num_cidades = len(cidades)
        self.num_formigas = num_formigas
        self.num_iteracoes = num_iteracoes
        self.alfa = alfa      # importância do feromônio
        self.beta = beta      # importância da heurística (visibilidade)
        self.rho = rho        # taxa de evaporação
        self.Q = Q            # quantidade de feromônio depositada

        # Matriz de feromônios inicial
        self.feromonios = np.ones((self.num_cidades, self.num_cidades))

        # Calcula matriz de heurísticas (1/distância)
        self.heuristicas = self._calcular_matriz_heuristicas()

        # Histórico para plotar a convergência
        self.historico_melhores = []

    # -------------------------------------------------------------
    # MATRIZ DE HEURÍSTICAS
    # -------------------------------------------------------------
    def _calcular_matriz_heuristicas(self):
        heuristicas = np.zeros((self.num_cidades, self.num_cidades))
        
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                dist = utils.calcular_distancia(self.cidades[i], self.cidades[j])
                valor = 1.0 / (dist + 1e-10)
                heuristicas[i][j] = valor
                heuristicas[j][i] = valor

        return heuristicas

    # -------------------------------------------------------------
    # EXECUÇÃO DO ACO
    # -------------------------------------------------------------
    def executar(self):
        print(f"Iniciando ACO: Formigas={self.num_formigas}, Iterações={self.num_iteracoes}")
        
        melhor_rota_global = None
        menor_distancia_global = float('inf')

        for i in range(1, self.num_iteracoes + 1):

            rotas_das_formigas = []

            # Constrói solução para cada formiga
            for _ in range(self.num_formigas):
                rota = self._construir_solucao()
                rotas_das_formigas.append(rota)

            # Atualiza feromônios
            self._atualizar_feromonio(rotas_das_formigas)

            # Avalia as rotas
            melhor_dist_iteracao = float('inf')

            for rota in rotas_das_formigas:
                d = utils.calcular_distancia_total(rota, self.cidades)

                if d < melhor_dist_iteracao:
                    melhor_dist_iteracao = d

                if d < menor_distancia_global:
                    menor_distancia_global = d
                    melhor_rota_global = rota

            # Salva histórico para gráfico
            self.historico_melhores.append(menor_distancia_global)

            # Log a cada 10 iterações
            if i % 10 == 0 or i == self.num_iteracoes:
                print(f"Iteração {i:3}: Melhor Iteração = {melhor_dist_iteracao:.2f} | Melhor Global = {menor_distancia_global:.2f}")

        print("Evolução do ACO concluída.")
        return melhor_rota_global, menor_distancia_global

    # -------------------------------------------------------------
    # CONSTRUÇÃO DA ROTA
    # -------------------------------------------------------------
    def _construir_solucao(self):
        cidade_inicial = random.randint(0, self.num_cidades - 1)
        rota = [cidade_inicial]

        cidades_a_visitar = set(range(self.num_cidades))
        cidades_a_visitar.remove(cidade_inicial)

        # Formiga constrói rota por probabilidade
        while cidades_a_visitar:
            atual = rota[-1]

            probabilidades = []
            cidades_possiveis = []

            for prox in cidades_a_visitar:
                f = self.feromonios[atual][prox]
                h = self.heuristicas[atual][prox]

                prob = (f ** self.alfa) * (h ** self.beta)
                probabilidades.append(prob)
                cidades_possiveis.append(prox)

            soma = sum(probabilidades)
            probs_norm = [p / (soma + 1e-10) for p in probabilidades]

            proxima = random.choices(cidades_possiveis, weights=probs_norm, k=1)[0]

            rota.append(proxima)
            cidades_a_visitar.remove(proxima)

        return rota

    # -------------------------------------------------------------
    # ATUALIZAÇÃO DE FEROMÔNIOS
    # -------------------------------------------------------------
    def _atualizar_feromonio(self, rotas_das_formigas):

        # Evaporação
        self.feromonios *= (1 - self.rho)

        # Depósito
        for rota in rotas_das_formigas:
            distancia = utils.calcular_distancia_total(rota, self.cidades)
            deposito = self.Q / distancia

            for i in range(self.num_cidades):
                a = rota[i]
                b = rota[(i + 1) % self.num_cidades]

                self.feromonios[a][b] += deposito
                self.feromonios[b][a] += deposito
