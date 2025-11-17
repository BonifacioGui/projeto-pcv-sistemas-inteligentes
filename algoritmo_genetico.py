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