import math
import random

# --- FUNÇÃO DE CÁLCULO DE DISTÂNCIA (EUCLIDIANA) ---

def calcular_distancia(cidadeA, cidadeB):
    """
    Calcula a distância Euclidiana entre duas cidades (pontos).
    cidadeA e cidadeB são tuplas (x, y).
    """ 

    # Fórmula: raiz_quadrada((x2 - x1)^2 + (y2 - y1)^2)
    return math.sqrt((cidadeB[0] - cidadeA[0])**2 + (cidadeB[1] - cidadeA[1])**2)


# --- FUNÇÃO DE APTIDÃO (FITNESS FUNCTION) ---

def calcular_distancia_total(rota, cidades):
    """
    Calcula a distância total de uma rota (permutação de cidades).
    Esta é a nossa principal FUNÇÃO DE APTIDÃO (Fitness Function).
    
    'rota' é uma lista de IDs (índices), ex: [0, 4, 2, 1, 3]
    'cidades' é a lista de tuplas (x, y) que você carregou do parser.
    """
    distancia_total = 0
    
    # Itera pela rota, somando a distância entre cidades consecutivas
    for i in range(len(rota)):
        
        # Pega as coordenadas da cidade atual
        cidade_atual = cidades[rota[i]]
        
        # Pega a próxima cidade na rota.
        # Se for a última cidade, a próxima é a cidade INICIAL.
        if i == len(rota) - 1:
            proxima_cidade = cidades[rota[0]] # Volta à origem
        else:
            proxima_cidade = cidades[rota[i+1]]
            
        # Acumula a distância
        distancia_total += calcular_distancia(cidade_atual, proxima_cidade)
        
    return distancia_total


# --- FUNÇÃO AUXILIAR DO ALGORITMO GENÉTICO ---

def criar_rota_aleatoria(num_cidades):
    """
    Cria uma rota aleatória (um indivíduo) para a população inicial.
    Garante que cada cidade seja visitada exatamente uma vez.
    """
    # 1. Cria uma lista ordenada: [0, 1, 2, ..., 69]
    rota = list(range(num_cidades))
    
    # 2. Embaralha a lista aleatoriamente
    random.shuffle(rota)
    
    # 3. Retorna a lista embaralhada, ex: [42, 1, 30, ...]
    return rota