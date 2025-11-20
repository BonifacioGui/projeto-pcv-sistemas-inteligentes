"""
Colônia de Formigas (ACO) - versão robusta e otimizada (MMAS-like)
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
        self.cidades = cidades
        self.num_cidades = len(cidades)

        self.num_formigas = int(min(self.num_cidades, 20)) if num_formigas is None else max(1, int(num_formigas))
        self.num_iteracoes = max(1, int(num_iteracoes))

        self.alfa = float(alfa)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)

        # matriz de distâncias
        self.dist_matrix = utils.calcular_matriz_distancias(self.cidades)

        # heurística (1 / dist)
        self.heuristicas = self._calcular_matriz_heuristicas()

        # feromônios
        avg_dist = self._estimativa_distancia_media()
        if pheromone_init is None:
            tau0 = 1.0 / (max(1.0, self.num_cidades) * max(avg_dist, EPS))
        else:
            tau0 = float(pheromone_init)
        self.pheromone_init = float(tau0)

        self.feromonios = np.full((self.num_cidades, self.num_cidades), self.pheromone_init, dtype=float)

        if pheromone_max is None:
            self.pheromone_max = max(1.0, self.pheromone_init * 10.0)
        else:
            self.pheromone_max = float(pheromone_max)

        if pheromone_min is None:
            self.pheromone_min = max(EPS, self.pheromone_max / (2.0 * self.num_cidades))
        else:
            self.pheromone_min = float(pheromone_min)

        self.elite_weight = float(elite_weight)
        self.stagnation_limit = max(1, int(stagnation_limit))
        self.historico_melhores: List[float] = []
        self.historico_max_length = historico_max_length or self.num_iteracoes

        self.best_route: Optional[List[int]] = None
        self.best_distance: float = float("inf")

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

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
        self.feromonios = 0.5 * self.feromonios + 0.5 * self.pheromone_init
        self._clip_pheromones()

    def _construir_solucao(self, start: Optional[int] = None) -> List[int]:
        if start is None:
            current = random.randrange(self.num_cidades)
        else:
            current = int(start)
        rota = [current]
        nao_visitadas = set(range(self.num_cidades))
        nao_visitadas.remove(current)

        fer = self.feromonios
        heur = self.heuristicas
        alfa = self.alfa
        beta = self.beta

        while nao_visitadas:
            cand = list(nao_visitadas)
            
            # CORREÇÃO DE BLINDAGEM AQUI
            taus = fer[current, cand]
            etas = heur[current, cand]

            # 1. Clip preventivo para evitar overflow na exponenciação
            # Limita valores absurdamente altos ou baixos antes de elevar a alfa/beta
            taus = np.clip(taus, 1e-15, 1e100)
            etas = np.clip(etas, 1e-15, 1e100)
            
            try:
                # Tenta calcular pesos
                pesos = (taus ** alfa) * (etas ** beta)
            except (FloatingPointError, OverflowError):
                # Se der erro matemático, assume pesos iguais (decisão aleatória)
                pesos = np.ones(len(cand))

            soma = float(pesos.sum())
            
            # 2. Validação da soma
            if soma <= 0.0 or not np.isfinite(soma):
                # Fallback: escolha aleatória uniforme se os pesos forem inválidos
                proxima = random.choice(cand)
            else:
                probs = pesos / soma
                # Escolha ponderada
                # np.random.choice pode dar erro se a soma das probs não for exatamente 1.0 (erro de float)
                # então normalizamos novamente ou usamos random.choices do python puro que é mais tolerante
                proxima = random.choices(cand, weights=probs, k=1)[0]

            rota.append(proxima)
            nao_visitadas.remove(proxima)
            current = proxima

        return rota

    def _atualizar_feromonio(self, rotas: List[List[int]]) -> None:
        self.feromonios *= (1.0 - self.rho)
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

        if self.best_route is not None and self.best_distance < float("inf"):
            extra = (self.Q / max(self.best_distance, EPS)) * self.elite_weight
            for k in range(len(self.best_route)):
                a = self.best_route[k]
                b = self.best_route[(k + 1) % len(self.best_route)]
                self.feromonios[a, b] += extra
                self.feromonios[b, a] += extra

        self._clip_pheromones()

    def executar(self) -> Tuple[List[int], float]:
        print(f"\n[ACO] Iniciando | Formigas={self.num_formigas} | Iter={self.num_iteracoes}")
        stagnation_counter = 0
        self.best_route = None
        self.best_distance = float("inf")
        self.historico_melhores = []

        for it in range(1, self.num_iteracoes + 1):
            rotas = []
            distancias = []
            for _ in range(self.num_formigas):
                rota = self._construir_solucao()
                rotas.append(rota)
                d = utils.calcular_distancia_total(rota, self.cidades, self.dist_matrix)
                distancias.append(d)

            idx_min = int(np.argmin(distancias))
            melhor_dist_iter = distancias[idx_min]
            melhor_rota_iter = rotas[idx_min]

            if melhor_dist_iter + EPS < self.best_distance:
                self.best_distance = float(melhor_dist_iter)
                self.best_route = list(melhor_rota_iter)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            self._atualizar_feromonio(rotas)
            self.historico_melhores.append(self.best_distance)
            
            if stagnation_counter >= self.stagnation_limit:
                # print(f"[ACO] Estagnação (it={it}). Reiniciando feromônios.")
                self._reinicializar_parcial()
                stagnation_counter = 0

            if it % 50 == 0 or it == self.num_iteracoes:
                print(f"[ACO] Iter {it:4}: Melhor_it={melhor_dist_iter:.4f} | Global={self.best_distance:.4f}")

        return (self.best_route or [], self.best_distance)