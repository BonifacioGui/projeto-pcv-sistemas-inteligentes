# colonia_formigas.py
"""
Colônia de Formigas (ACO) - versão robusta e otimizada (MMAS-like)

Principais características:
- usa matriz de distâncias pré-computada (muito mais rápido)
- feromônios com limites (tau_min / tau_max) estilo MMAS
- depósito elitista (melhor global reforça mais)
- detecção de estagnação e reinicialização parcial de feromônios
- histórico de melhores por iteração
- implementação vetorizada onde apropriado (numpy)
- proteções numéricas (EPS) e fallbacks
"""
from typing import List, Tuple, Optional
import numpy as np
import random

import utils

EPS = 1e-12


class ACO:
    def __init__(
        self,
        cidades: List[Tuple[float, float]],
        num_formigas: Optional[int] = None,
        num_iteracoes: int = 200,
        alfa: float = 1.0,
        beta: float = 2.5,
        rho: float = 0.1,
        Q: float = 1.0,
        pheromone_init: Optional[float] = None,
        pheromone_min: Optional[float] = None,
        pheromone_max: Optional[float] = None,
        elite_weight: float = 2.0,
        stagnation_limit: int = 50,
        historico_max_length: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ):
        """
        Parâmetros:
          - cidades: lista de (x,y)
          - num_formigas: se None, usa min(num_cidades, 20)
          - num_iteracoes: iterações do algoritmo
          - alfa, beta: pesos de feromônio e heurística
          - rho: taxa de evaporação (0..1)
          - Q: constante para cálculo de depósito (normalmente 1.0 ou 100.0)
          - pheromone_init/min/max: limites/valor inicial. Se None, valores sensatos são estimados.
          - elite_weight: multiplicador para depósito da melhor rota global
          - stagnation_limit: reinicializa parcialmente feromônios quando não houver melhora
          - historico_max_length: limita o histórico guardado (None = sem limite)
          - rng_seed: semente para reprodutibilidade
        """
        self.cidades = cidades
        self.num_cidades = len(cidades)

        self.num_formigas = int(min(self.num_cidades, 20)) if num_formigas is None else max(1, int(num_formigas))
        self.num_iteracoes = max(1, int(num_iteracoes))

        self.alfa = float(alfa)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)

        # matriz de distâncias (lista de listas) — usar em cálculos de distância
        self.dist_matrix = utils.calcular_matriz_distancias(self.cidades)

        # heurística (1 / dist), evitando divisão por zero
        self.heuristicas = self._calcular_matriz_heuristicas()

        # feromônios: valor inicial sensato
        # se não informado, estimamos um tau0 = 1 / (n * avg_dist)
        avg_dist = self._estimativa_distancia_media()
        if pheromone_init is None:
            tau0 = 1.0 / (max(1.0, self.num_cidades) * max(avg_dist, EPS))
        else:
            tau0 = float(pheromone_init)
        self.pheromone_init = float(tau0)

        # inicializa matriz de feromônio
        self.feromonios = np.full((self.num_cidades, self.num_cidades), self.pheromone_init, dtype=float)

        # limites (tau_min, tau_max). Se não passados, definimos de forma adaptativa.
        # tau_max = 1 / (rho * best_dist) é uma fórmula comum, mas sem best_dist inicial usamos heurísticas
        if pheromone_max is None:
            # usa um valor grande relativo a tau0; será ajustado após encontrar uma boa solução
            self.pheromone_max = max(1.0, self.pheromone_init * 10.0)
        else:
            self.pheromone_max = float(pheromone_max)

        if pheromone_min is None:
            # default conservador
            self.pheromone_min = max(EPS, self.pheromone_max / (2.0 * self.num_cidades))
        else:
            self.pheromone_min = float(pheromone_min)

        # elitismo / rank deposit
        self.elite_weight = float(elite_weight)

        # estagnação: se não houver melhora por 'stagnation_limit' iterações, reinicializa parcialmente
        self.stagnation_limit = max(1, int(stagnation_limit))

        # histórico
        self.historico_melhores: List[float] = []
        self.historico_max_length = historico_max_length or self.num_iteracoes

        # estado de melhor global
        self.best_route: Optional[List[int]] = None
        self.best_distance: float = float("inf")

        # semente
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

    # -----------------------
    # utilitários
    # -----------------------
    def _calculcar_dist_matrix_numpy(self) -> np.ndarray:
        """Retorna a matriz de distâncias como numpy array (copiada da dist_matrix)."""
        return np.array(self.dist_matrix, dtype=float)

    def _calcular_matriz_heuristicas(self) -> np.ndarray:
        h = np.zeros((self.num_cidades, self.num_cidades), dtype=float)
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                d = self.dist_matrix[i][j]
                h_val = 1.0 / (d + EPS)
                h[i, j] = h_val
                h[j, i] = h_val
        return h

    def _estimativa_distancia_media(self) -> float:
        # média das distâncias fora diagonal
        total = 0.0
        count = 0
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                total += self.dist_matrix[i][j]
                count += 1
        return (total / count) if count > 0 else 1.0

    def _clip_pheromones(self) -> None:
        np.clip(self.feromonios, self.pheromone_min, self.pheromone_max, out=self.feromonios)

    def _reinicializar_parcial(self) -> None:
        """
        Blend parcialmente os feromônios com o valor inicial para sair de estagnação.
        Isso reduz a influência acumulada sem apagar tudo.
        """
        self.feromonios = 0.5 * self.feromonios + 0.5 * self.pheromone_init
        self._clip_pheromones()

    # -----------------------
    # Construção de solução (uma formiga)
    # -----------------------
    def _construir_solucao(self, start: Optional[int] = None) -> List[int]:
        """
        Constrói uma rota (lista de índices).
        Usa a regra probabilística com feromônio^alfa * heuristica^beta.
        """
        if start is None:
            current = random.randrange(self.num_cidades)
        else:
            current = int(start)
        rota = [current]
        nao_visitadas = set(range(self.num_cidades))
        nao_visitadas.remove(current)

        # para acelerar, referenciar arrays locais
        fer = self.feromonios
        heur = self.heuristicas
        alfa = self.alfa
        beta = self.beta

        while nao_visitadas:
            cand = list(nao_visitadas)
            # computa pesos: tau^alfa * eta^beta
            taus = fer[current, cand]  # numpy fancy indexing
            etas = heur[current, cand]
            # calcula pesos de modo numérico estável
            # evita potenciação de números muito pequenos/grandes usando np.power direto
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                try:
                    pesos = np.power(taus, alfa) * np.power(etas, beta)
                except FloatingPointError:
                    # fallback: normaliza taus e etas para evitar overflow/underflow
                    taus_safe = np.clip(taus, EPS, None)
                    etas_safe = np.clip(etas, EPS, None)
                    pesos = np.power(taus_safe, alfa) * np.power(etas_safe, beta)

            soma = float(pesos.sum())
            if soma <= 0.0 or not np.isfinite(soma):
                # fallback: escolha aleatória uniforme
                proxima = random.choice(cand)
            else:
                # escolha com prob proporcional às pesos
                probs = pesos / soma
                # random.choice via numpy para vetor
                proxima = int(np.random.choice(cand, p=probs))
            rota.append(proxima)
            nao_visitadas.remove(proxima)
            current = proxima

        return rota

    # -----------------------
    # Atualização de feromônio
    # -----------------------
    def _atualizar_feromonio(self, rotas: List[List[int]]) -> None:
        """
        1) Evaporação
        2) Depósito: deposit_all (todas as formigas) + extra para best_global (elitismo)
        Depois aplica clipping (tau_min, tau_max).
        """
        # Evaporação (vetorizada)
        self.feromonios *= (1.0 - self.rho)

        # Depósito por cada formiga (vectorized-ish)
        for rota in rotas:
            dist = utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)
            if dist <= 0 or not np.isfinite(dist):
                continue
            delta = (self.Q / dist)
            # adiciona delta em cada aresta da rota (bidirecional)
            for k in range(len(rota)):
                i = rota[k]
                j = rota[(k + 1) % len(rota)]
                self.feromonios[i, j] += delta
                self.feromonios[j, i] += delta

        # Depósito extra (elitismo) pela melhor rota global encontrada
        if self.best_route is not None and self.best_distance < float("inf"):
            extra = (self.Q / max(self.best_distance, EPS)) * self.elite_weight
            for k in range(len(self.best_route)):
                a = self.best_route[k]
                b = self.best_route[(k + 1) % len(self.best_route)]
                self.feromonios[a, b] += extra
                self.feromonios[b, a] += extra

        # Limita feromônios (MMAS style)
        self._clip_pheromones()

    # -----------------------
    # Execução principal
    # -----------------------
    def executar(self) -> Tuple[List[int], float]:
        """
        Executa o ACO completo e retorna (melhor_rota, melhor_distancia).
        Também popula self.historico_melhores com o melhor global por iteração.
        """
        print(f"\n[ACO] Iniciando | Formigas={self.num_formigas} | Iter={self.num_iteracoes} | alfa={self.alfa} beta={self.beta} rho={self.rho}")
        stagnation_counter = 0
        self.best_route = None
        self.best_distance = float("inf")
        self.historico_melhores = []

        for it in range(1, self.num_iteracoes + 1):
            rotas = []
            distancias = []

            # construir soluções
            for _ in range(self.num_formigas):
                rota = self._construir_solucao()
                rotas.append(rota)
                d = utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)
                distancias.append(d)

            # encontra melhor da iteração
            idx_min = int(np.argmin(distancias))
            melhor_dist_iter = distancias[idx_min]
            melhor_rota_iter = rotas[idx_min]

            # atualiza melhor global
            if melhor_dist_iter + EPS < self.best_distance:
                self.best_distance = float(melhor_dist_iter)
                self.best_route = list(melhor_rota_iter)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # atualiza feromonios (evaporação + depósitos + elitismo)
            self._atualizar_feromonio(rotas)

            # guarda histórico
            self.historico_melhores.append(self.best_distance)
            if len(self.historico_melhores) > self.historico_max_length:
                self.historico_melhores = self.historico_melhores[-self.historico_max_length :]

            # Detecta estagnação e aplica reinicialização parcial de feromônio
            if stagnation_counter >= self.stagnation_limit:
                print(f"[ACO] Estagnação detectada (it={it}). Reinicializando parcialmente feromônios.")
                self._reinicializar_parcial()
                stagnation_counter = 0

            # Log periódico
            if it % 10 == 0 or it == self.num_iteracoes:
                print(f"[ACO] Iter {it:4}: Melhor_it={melhor_dist_iter:.4f} | Melhor_global={self.best_distance:.4f}")

        print("[ACO] Concluído.")
        return (self.best_route or [], self.best_distance)
