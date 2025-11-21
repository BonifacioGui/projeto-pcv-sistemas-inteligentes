"""
Colônia de Formigas (ACO) - versão robusta e otimizada (MMAS-like)
"""

from typing import List, Tuple, Optional
import numpy as np
import random
import utils

# Pequena constante numérica para evitar divisões por zero
EPS = 1e-12


class ACO:
    """
    Implementação robusta de ACO, com elementos inspirados no MMAS (Max-Min Ant System):
    - feromônio limitado (min/max)
    - reforço elitista
    - reinicialização parcial ao detectar estagnação
    """

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
        Inicializa os parâmetros do ACO.

        PARTE DIDÁTICA (para a apresentação):
        -------------------------------------
        - alfa controla a influência do feromônio
        - beta controla a influência da heurística (1/distância)
        - rho é a taxa de evaporação
        - num_formigas ≈ nº cidades, por padrão
        - limite de estagnação detecta quando o algoritmo “parou de aprender”
        """

        # Dados do problema
        self.cidades = cidades
        self.num_cidades = len(cidades)

        # Número de formigas padrão (min(cidades, 20))
        self.num_formigas = int(min(self.num_cidades, 20)) if num_formigas is None else max(1, int(num_formigas))
        self.num_iteracoes = max(1, int(num_iteracoes))

        # Hiperparâmetros do ACO
        self.alfa = float(alfa)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)

        # Matriz de distâncias pré-calculada (grande otimização)
        self.dist_matrix = utils.calcular_matriz_distancias(self.cidades)

        # Matriz heurística η = 1/d
        self.heuristicas = self._calcular_matriz_heuristicas()

        # Estimativa da distância média (usada para inicializar τ0)
        avg_dist = self._estimativa_distancia_media()
        if pheromone_init is None:
            # Fórmula recomendada em literatura de MMAS
            tau0 = 1.0 / (max(1.0, self.num_cidades) * max(avg_dist, EPS))
        else:
            tau0 = float(pheromone_init)
        self.pheromone_init = float(tau0)

        # Matriz de feromônio inicial
        self.feromonios = np.full((self.num_cidades, self.num_cidades), self.pheromone_init, dtype=float)

        # Limites do feromônio
        # (Muito importantes para evitar explosão numérica)
        if pheromone_max is None:
            self.pheromone_max = max(1.0, self.pheromone_init * 10.0)
        else:
            self.pheromone_max = float(pheromone_max)

        if pheromone_min is None:
            self.pheromone_min = max(EPS, self.pheromone_max / (2.0 * self.num_cidades))
        else:
            self.pheromone_min = float(pheromone_min)

        # Reforço elitista
        self.elite_weight = float(elite_weight)

        # Controle de estagnação
        self.stagnation_limit = max(1, int(stagnation_limit))
        self.historico_melhores: List[float] = []
        self.historico_max_length = historico_max_length or self.num_iteracoes

        # Melhor solução global
        self.best_route: Optional[List[int]] = None
        self.best_distance: float = float("inf")

        # Reprodutibilidade
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

    # ======================================================================
    # MATRIZES AUXILIARES
    # ======================================================================
    def _calcular_matriz_heuristicas(self) -> np.ndarray:
        """
        Cria matriz η_ij = 1/d_ij.
        (Heurística clássica do ACO para o TSP)
        """
        h = np.zeros((self.num_cidades, self.num_cidades), dtype=float)
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                d = self.dist_matrix[i][j]
                h_val = 1.0 / (d + EPS)
                h[i, j] = h_val
                h[j, i] = h_val
        return h

    def _estimativa_distancia_media(self) -> float:
        """Usada apenas para calibrar τ0 (valor inicial do feromônio)."""
        total = 0.0
        count = 0
        for i in range(self.num_cidades):
            for j in range(i + 1, self.num_cidades):
                total += self.dist_matrix[i][j]
                count += 1
        return (total / count) if count > 0 else 1.0

    # ======================================================================
    # GESTÃO DO FEROMÔNIO
    # ======================================================================
    def _clip_pheromones(self) -> None:
        """Garante τ_min ≤ τ ≤ τ_max (MMAS clássico)."""
        np.clip(self.feromonios, self.pheromone_min, self.pheromone_max, out=self.feromonios)

    def _reinicializar_parcial(self) -> None:
        """
        Reaproveita parte do feromônio e mistura com τ0.
        Usado quando o algoritmo estagna.
        """
        self.feromonios = 0.5 * self.feromonios + 0.5 * self.pheromone_init
        self._clip_pheromones()

    # ======================================================================
    # CONSTRUÇÃO DAS ROTAS (com proteção contra overflow numérico)
    # ======================================================================
    def _construir_solucao(self, start: Optional[int] = None) -> List[int]:
        """
        Constrói uma rota completa aplicando a regra de probabilidade do ACO.

        Parte crítica:
        --------------
        Aqui incluímos **blindagem numérica** para evitar overflow
        quando elevamos τ^α e η^β.
        """

        current = random.randrange(self.num_cidades) if start is None else int(start)
        rota = [current]

        nao_visitadas = set(range(self.num_cidades))
        nao_visitadas.remove(current)

        fer = self.feromonios
        heur = self.heuristicas
        alfa = self.alfa
        beta = self.beta

        while nao_visitadas:
            cand = list(nao_visitadas)

            # Seleciona os valores τ e η
            taus = fer[current, cand]
            etas = heur[current, cand]

            # ----- BLINDAGEM NUMÉRICA -----
            # Evita valores extremamente pequenos ou extremamente grandes
            taus = np.clip(taus, 1e-15, 1e100)
            etas = np.clip(etas, 1e-15, 1e100)

            try:
                # Cálculo dos pesos
                pesos = (taus ** alfa) * (etas ** beta)
            except (FloatingPointError, OverflowError):
                # fallback seguro
                pesos = np.ones(len(cand))

            soma = float(pesos.sum())

            # Validação da soma antes de normalizar
            if soma <= 0.0 or not np.isfinite(soma):
                # fallback para escolha aleatória
                proxima = random.choice(cand)
            else:
                # Escolha probabilística
                probs = pesos / soma
                # python.nativo tem mais tolerância com floats imprecisos
                proxima = random.choices(cand, weights=probs, k=1)[0]

            rota.append(proxima)
            nao_visitadas.remove(proxima)
            current = proxima

        return rota

    # ======================================================================
    # ATUALIZAÇÃO DO FEROMÔNIO
    # ======================================================================
    def _atualizar_feromonio(self, rotas: List[List[int]]) -> None:
        """
        Atualiza o feromônio aplicando:
        - evaporação
        - reforço das formigas da iteração
        - reforço elitista da melhor solução global
        """
        # Evaporação
        self.feromonios *= (1.0 - self.rho)

        # Deposição das formigas
        for rota in rotas:
            dist = utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)
            if dist <= 0 or not np.isfinite(dist):
                continue
            delta = (self.Q / dist)
            for k in range(len(rota)):
                i = rota[k]
                j = rota[(k + 1) % len(rota)]
                self.feromonios[i, j] += delta
                self.feromonios[j, i] += delta

        # Reforço elitista
        if self.best_route is not None and self.best_distance < float("inf"):
            extra = (self.Q / max(self.best_distance, EPS)) * self.elite_weight
            for k in range(len(self.best_route)):
                a = self.best_route[k]
                b = self.best_route[(k + 1) % len(self.best_route)]
                self.feromonios[a, b] += extra
                self.feromonios[b, a] += extra

        # Garante limites do MMAS
        self._clip_pheromones()

    # ======================================================================
    # EXECUÇÃO PRINCIPAL DO ACO
    # ======================================================================
    def executar(self) -> Tuple[List[int], float]:
        print(f"\n[ACO] Iniciando | Formigas={self.num_formigas} | Iter={self.num_iteracoes}")
        stagnation_counter = 0
        self.best_route = None
        self.best_distance = float("inf")
        self.historico_melhores = []

        for it in range(1, self.num_iteracoes + 1):
            rotas = []
            distancias = []

            # Cada formiga constrói sua rota independentemente
            for _ in range(self.num_formigas):
                rota = self._construir_solucao()
                rotas.append(rota)

                d = utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)
                distancias.append(d)

            # Melhor formiga da iteração
            idx_min = int(np.argmin(distancias))
            melhor_dist_iter = distancias[idx_min]
            melhor_rota_iter = rotas[idx_min]

            # Atualiza melhor global
            if melhor_dist_iter + EPS < self.best_distance:
                self.best_distance = float(melhor_dist_iter)
                self.best_route = list(melhor_rota_iter)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Atualização global dos feromônios
            self._atualizar_feromonio(rotas)
            self.historico_melhores.append(self.best_distance)

            # Detecta estagnação e faz reset parcial
            if stagnation_counter >= self.stagnation_limit:
                self._reinicializar_parcial()
                stagnation_counter = 0

            # Log periódico
            if it % 50 == 0 or it == self.num_iteracoes:
                print(f"[ACO] Iter {it:4}: Melhor_it={melhor_dist_iter:.4f} | Global={self.best_distance:.4f}")

        # Retorna melhor rota e melhor distância final
        return (self.best_route or [], self.best_distance)
