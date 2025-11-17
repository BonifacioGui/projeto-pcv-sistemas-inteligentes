# Dentro de recozimento_simulado.py

import random
import utils
import math

class RecozimentoSimulado:
    """
    Implementa o algoritmo Simulated Annealing (SA) para o PCV.
    """
    def __init__(self, cidades, temp_inicial, temp_final, taxa_resfriamento):
        
        # 1. Dados do Problema
        self.cidades = cidades
        self.num_cidades = len(cidades)
        
        # 2. Hiperparâmetros do SA
        self.temp_inicial = temp_inicial
        self.temp_final = temp_final
        self.taxa_resfriamento = taxa_resfriamento
        
        # 3. Estado Atual
        self.solucao_atual = None
        self.distancia_atual = float('inf')
        
        # 4. Melhor Solução Encontrada
        self.melhor_solucao_global = None
        self.melhor_distancia_global = float('inf')
        
        # 5. Histórico para o gráfico de convergência
        self.historico_melhores = []

    def _gerar_solucao_vizinha(self, rota):
        """
        Gera uma solução "vizinha" trocando duas cidades.
        (Basicamente, uma mutação por troca 'sempre ligada')
        """
        rota_vizinha = rota[:] # Copia a rota
        
        # Sorteia dois índices para trocar
        idx1, idx2 = random.sample(range(self.num_cidades), 2)
        
        # Realiza a troca (swap)
        rota_vizinha[idx1], rota_vizinha[idx2] = rota_vizinha[idx2], rota_vizinha[idx1]
        
        return rota_vizinha

    def executar(self):
        """
        Executa o loop principal do Recozimento Simulado.
        """
        print(f"Iniciando SA: Temp. Inicial={self.temp_inicial}, Taxa Resfriamento={self.taxa_resfriamento}")

        # 1. Gera a solução inicial (aleatória)
        self.solucao_atual = utils.criar_rota_aleatoria(self.num_cidades)
        self.distancia_atual = utils.calcular_distancia_total(self.solucao_atual, self.cidades)
        
        # Define a solução inicial como a melhor global por enquanto
        self.melhor_solucao_global = self.solucao_atual
        self.melhor_distancia_global = self.distancia_atual
        
        temperatura = self.temp_inicial
        
        # 2. Loop principal (enquanto o sistema está "quente")
        while temperatura > self.temp_final:
            
            # a. Gera um vizinho
            solucao_vizinha = self._gerar_solucao_vizinha(self.solucao_atual)
            distancia_vizinha = utils.calcular_distancia_total(solucao_vizinha, self.cidades)
            
            # b. Calcula a diferença de custo (Delta E)
            #    Se delta < 0, o vizinho é MELHOR (distância menor)
            #    Se delta > 0, o vizinho é PIOR (distância maior)
            delta_distancia = distancia_vizinha - self.distancia_atual
            
            # c. Decide se aceita o vizinho
            if delta_distancia < 0:
                # Movimento bom: Sempre aceita
                self.solucao_atual = solucao_vizinha
                self.distancia_atual = distancia_vizinha
            else:
                # Movimento ruim: Aceita com uma probabilidade
                # 
                prob_aceitacao = math.exp(-delta_distancia / temperatura)
                if random.random() < prob_aceitacao:
                    self.solucao_atual = solucao_vizinha
                    self.distancia_atual = distancia_vizinha
            
            # d. Atualiza o melhor global, se necessário
            if self.distancia_atual < self.melhor_distancia_global:
                self.melhor_solucao_global = self.solucao_atual
                self.melhor_distancia_global = self.distancia_atual
            
            # e. Resfria a temperatura
            temperatura *= self.taxa_resfriamento
            
            # f. Salva no histórico para o gráfico
            self.historico_melhores.append(self.melhor_distancia_global)

        print("Recozimento (SA) concluído.")
        print(f"Melhor distância final: {self.melhor_distancia_global:.2f}")
        return self.melhor_solucao_global, self.melhor_distancia_global