# recozimento_simulado.py
"""
Simulated Annealing (SA) otimizado para TSP.

Principais características:
- Uso opcional de dist_matrix (pré-calculada) para ganho de velocidade.
- Inicialização por heurística nearest-neighbor.
- Vizinhança 2-opt (reversão de segmento) com avaliação incremental.
- Ciclo por temperatura: realiza `moves_per_temp` tentativas antes de resfriar.
- Registro de histórico com amostragem para evitar listas gigantes.
- Parâmetros configuráveis: temp_inicial, temp_final, cooling_rate, moves_per_temp, max_iterations.
"""

from typing import List, Tuple, Optional
import math
import random

import utils  # suas funções otimizadas (calcular_distancia_total / criar_rota_aleatoria / criar_matriz_distancias)


class RecozimentoSimulado:
    def __init__(
        self,
        cidades: List[Tuple[float, float]],
        temp_inicial: float = 1000.0,
        temp_final: float = 1e-3,
        cooling_rate: float = 0.995,
        moves_per_temp: Optional[int] = None,
        max_iterations: Optional[int] = None,
        use_2opt: bool = True,
        dist_matrix: Optional[List[List[float]]] = None,
        record_interval: int = 10,
        rng_seed: Optional[int] = None,
    ):
        """
        Parâmetros:
          - cidades: lista de (x,y)
          - temp_inicial, temp_final: temperaturas inicial/final
          - cooling_rate: fator multiplicativo por etapa (0 < cooling_rate < 1)
          - moves_per_temp: quantos vizinhos testar por temperatura (None -> n_cidades)
          - max_iterations: limite total de iterações (movimentos) como fallback
          - use_2opt: se True usa 2-opt, senão usa swap
          - dist_matrix: matriz de distâncias pré-computada (recomendada)
          - record_interval: salva histórico a cada N movimentos (evita arrays gigantes)
          - rng_seed: semente para reprodutibilidade
        """
        self.cidades = cidades
        self.n = len(cidades)
        self.temp_inicial = float(temp_inicial)
        self.temp_final = float(temp_final)
        self.cooling_rate = float(cooling_rate)
        self.moves_per_temp = moves_per_temp if moves_per_temp is not None else max(1, self.n)
        self.max_iterations = max_iterations
        self.use_2opt = use_2opt
        self.dist_matrix = dist_matrix
        self.record_interval = max(1, int(record_interval))

        # Estado
        self.solucao_atual: List[int] = []
        self.distancia_atual: float = float("inf")
        self.melhor_solucao_global: List[int] = []
        self.melhor_distancia_global: float = float("inf")
        self.historico_melhores: List[float] = []

        # RNG isolado (reprodutível)
        self._rng = random.Random(rng_seed)

    # -------------------------
    # Inicializadores / utilitários
    # -------------------------
    def _nearest_neighbor(self, start: int = 0) -> List[int]:
        """Heurística gulosa: nearest neighbor — retorna permutação inicial."""
        
        if self.n == 0:
            return []
        unvisited = set(range(self.n))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            # escolher o mais próximo entre unvisited
            best = None
            best_d = float("inf")
            for v in unvisited:
                d = self.dist_matrix[last][v] if self.dist_matrix is not None else utils.calcular_distancia(self.cidades[last], self.cidades[v])
                if d < best_d:
                    best_d = d
                    best = v
            tour.append(best)
            unvisited.remove(best)
        return tour

    def _calc_distance(self, rota: List[int]) -> float:
        """Usa dist_matrix quando disponível (muito mais rápido)."""
        return utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)

    # -------------------------
    # Operadores de vizinhança
    # -------------------------
    def _neighbor_swap(self, rota: List[int]) -> Tuple[List[int], int, int]:
        """Gera vizinho por swap e retorna (rota_mutada, i, j)."""
        i, j = self._rng.sample(range(self.n), 2)
        if i > j:
            i, j = j, i
        nova = rota[:]
        nova[i], nova[j] = nova[j], nova[i]
        return nova, i, j

    def _neighbor_2opt(self, rota: List[int]) -> Tuple[List[int], int, int]:
        """Gera vizinho por 2-opt (inverte segmento i..j) e retorna (rota_mutada, i, j) com i<j."""
        i = self._rng.randrange(0, self.n - 1)
        j = self._rng.randrange(i + 1, self.n)
        nova = rota[:i] + list(reversed(rota[i:j + 1])) + rota[j + 1:]
        return nova, i, j

    # -------------------------
    # Avaliação incremental 2-opt (opcional micro-otimização)
    # -------------------------
    def _delta_2opt(self, rota: List[int], i: int, j: int) -> float:
        """
        Calcula mudança de distância introduzida por 2-opt invertendo [i..j].
        Retorna nova_dist - old_dist (delta).
        Usa dist_matrix se disponível.
        """
        a, b = rota[i - 1], rota[i] if i > 0 else rota[-1]
        c, d = rota[j], rota[(j + 1) % self.n]

        if self.dist_matrix is not None:
            dm = self.dist_matrix
            old = dm[a][b] + dm[c][d]
            new = dm[a][c] + dm[b][d]
        else:
            old = utils.calcular_distancia(self.cidades[a], self.cidades[b]) + utils.calcular_distancia(self.cidades[c], self.cidades[d])
            new = utils.calcular_distancia(self.cidades[a], self.cidades[c]) + utils.calcular_distancia(self.cidades[b], self.cidades[d])

        return new - old

    def _delta_swap(self, rota: List[int], i: int, j: int) -> float:
        """
        Calcula delta para swap simples (troca i e j).
        Implementação direta (pode ser mais cara que delta_2opt).
        """
        # nodes involved: prev_i, i, next_i, prev_j, j, next_j (consider overlaps)
        n = self.n
        a_i = rota[i]
        a_j = rota[j]
        prev_i = rota[i - 1] if i > 0 else rota[-1]
        next_i = rota[(i + 1) % n]
        prev_j = rota[j - 1] if j > 0 else rota[-1]
        next_j = rota[(j + 1) % n]

        dm = self.dist_matrix
        if dm is not None:
            old = 0.0
            new = 0.0
            # remove edges prev_i - i and i - next_i (unless adjacent to j)
            if i + 1 == j:
                # adjacent case: prev_i - i - j - next_j
                old += dm[prev_i][a_i] + dm[a_i][a_j] + dm[a_j][next_j]
                new += dm[prev_i][a_j] + dm[a_j][a_i] + dm[a_i][next_j]
            else:
                old += dm[prev_i][a_i] + dm[a_i][next_i] + dm[prev_j][a_j] + dm[a_j][next_j]
                new += dm[prev_i][a_j] + dm[a_j][next_i] + dm[prev_j][a_i] + dm[a_i][next_j]
            return new - old
        else:
            # fallback: compute full distances
            old_total = 0.0
            new_total = 0.0
            # compute by enumerating affected edges (safe but slower)
            affected = set([prev_i, a_i, next_i, prev_j, a_j, next_j])
            # compute full old and new by direct distance sums (brute)
            # simpler to compute full route distances difference (slower)
            full_old = self._calc_distance(rota)
            rota_swap = rota[:]
            rota_swap[i], rota_swap[j] = rota_swap[j], rota_swap[i]
            full_new = self._calc_distance(rota_swap)
            return full_new - full_old

    # -------------------------
    # Execução principal
    # -------------------------
    def executar(self) -> Tuple[List[int], float]:
        """
        Executa o SA com Inicialização ALEATÓRIA (Academicamente Justo).
        """
        # --- CORREÇÃO ACADÊMICA: Inicialização Aleatória ---
        # Agora usamos criar_rota_aleatoria para garantir que o SA comece do zero.
        
        self.solucao_atual = utils.criar_rota_aleatoria(self.n)  # <--- CORRIGIDO
        self.distancia_atual = self._calc_distance(self.solucao_atual)
        
        # Define o melhor global inicial como a solução aleatória gerada
        self.melhor_solucao_global = self.solucao_atual[:]
        self.melhor_distancia_global = self.distancia_atual
        
        # Configuração inicial
        temperatura = float(self.temp_inicial)
        total_moves = 0
        self.historico_melhores = [self.melhor_distancia_global]

        # Loop principal (o resto do código permanece igual)
        while temperatura > self.temp_final:
            for _ in range(self.moves_per_temp):
                total_moves += 1

                # Gera vizinho (2-opt ou swap)
                if self.use_2opt:
                    viz, i, j = self._neighbor_2opt(self.solucao_atual)
                    delta = self._delta_2opt(self.solucao_atual, i, j)
                else:
                    viz, i, j = self._neighbor_swap(self.solucao_atual)
                    delta = self._delta_swap(self.solucao_atual, i, j)

                # Critério de Aceitação (Metropolis)
                if delta < 0:
                    self.solucao_atual = viz
                    self.distancia_atual += delta
                else:
                    try:
                        prob = math.exp(-delta / temperatura)
                    except OverflowError:
                        prob = 0.0
                    
                    if self._rng.random() < prob:
                        self.solucao_atual = viz
                        self.distancia_atual += delta

                # Atualiza o melhor global
                if self.distancia_atual < self.melhor_distancia_global:
                    self.melhor_distancia_global = self.distancia_atual
                    self.melhor_solucao_global = self.solucao_atual[:]

                # Histórico
                if total_moves % self.record_interval == 0:
                    self.historico_melhores.append(self.melhor_distancia_global)

                if self.max_iterations is not None and total_moves >= self.max_iterations:
                    break

            temperatura *= self.cooling_rate
            if self.max_iterations is not None and total_moves >= self.max_iterations:
                break

        # Gravação final
        if self.historico_melhores[-1] != self.melhor_distancia_global:
            self.historico_melhores.append(self.melhor_distancia_global)

        return self.melhor_solucao_global, self.melhor_distancia_global