import math
import random
import numpy as np

# ------------------------------------------------------------
# 1. DISTÂNCIA EUCLIDIANA OTIMIZADA
#    Usando math.hypot → implementado em C → muito mais rápido
# ------------------------------------------------------------

def calcular_distancia(cidadeA, cidadeB):
    """
    Calcula a distância Euclidiana entre duas cidades.
    cidadeA e cidadeB são tuplas (x, y).
    Usa math.hypot, que é otimizado em C.
    """
    return math.hypot(cidadeB[0] - cidadeA[0], cidadeB[1] - cidadeA[1])


# ------------------------------------------------------------
# 2. MATRIZ DE DISTÂNCIAS PRÉ-CALCULADA (para alta performance)
# ------------------------------------------------------------

def calcular_matriz_distancias(cidades):
    """
    Cria e retorna uma matriz NxN contendo todas as distâncias 
    pré-calculadas entre as cidades.
    
    ESSENCIAL para acelerar AG, ACO e SA.
    """

    num_cidades = len(cidades)
    matriz = np.zeros((num_cidades, num_cidades), dtype=float)

    for i in range(num_cidades):
        xi, yi = cidades[i]
        for j in range(i + 1, num_cidades):
            xj, yj = cidades[j]
            
            dist = math.hypot(xj - xi, yj - yi)
            matriz[i][j] = dist
            matriz[j][i] = dist  # simétrico

    return matriz


# ------------------------------------------------------------
# 3. DISTÂNCIA TOTAL DA ROTA (versão rápida usando matriz)
# ------------------------------------------------------------

def calcular_distancia_total(rota, cidades, dist_matrix=None):
    """
    Calcula a distância total de uma rota.

    OTIMIZAÇÃO:
       - Se dist_matrix for fornecida, usa distâncias pré-calculadas.
       - Caso contrário, faz o cálculo tradicional (mais lento).

    Parâmetros:
        rota        → lista de índices (ex: [0, 4, 2, ...])
        cidades     → lista de tuplas (x, y)
        dist_matrix → matriz NxN pré-calculada (opcional)
    """

    total = 0.0
    n = len(rota)

    # --- Caminho RÁPIDO: usando matriz de distâncias ---
    if dist_matrix is not None:
        for i in range(n):
            a = rota[i]
            b = rota[(i + 1) % n]  # ciclo fechado
            total += dist_matrix[a][b]
        return total

    # --- Caminho lento: cálculo bruto ---
    for i in range(n):
        cidadeA = cidades[rota[i]]
        cidadeB = cidades[rota[(i + 1) % n]]
        total += calcular_distancia(cidadeA, cidadeB)

    return total


# ------------------------------------------------------------
# 4. CRIA UMA ROTA ALEATÓRIA (permuta perfeitamente uniforme)
# ------------------------------------------------------------

def criar_rota_aleatoria(num_cidades):
    """
    Cria uma rota aleatória, garantindo que todas as cidades
    aparecem exatamente uma vez.
    """
    rota = list(range(num_cidades))
    random.shuffle(rota)
    return rota
# No final do arquivo utils.py

def gerar_rota_vizinho_mais_proximo(cidades, dist_matrix=None):
    """
    Gera uma rota gulosa (Vizinho Mais Próximo).
    Começa na cidade 0 e vai sempre para a mais próxima não visitada.
    """
    num_cidades = len(cidades)
    nao_visitadas = set(range(1, num_cidades))
    rota = [0] # Começa na cidade 0
    atual = 0

    while nao_visitadas:
        mais_proxima = -1
        min_dist = float('inf')

        for candidata in nao_visitadas:
            if dist_matrix is not None:
                d = dist_matrix[atual][candidata]
            else:
                d = calcular_distancia(cidades[atual], cidades[candidata])
            
            if d < min_dist:
                min_dist = d
                mais_proxima = candidata
        
        rota.append(mais_proxima)
        nao_visitadas.remove(mais_proxima)
        atual = mais_proxima
        
    return rota