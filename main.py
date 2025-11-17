# Dentro de main.py

import parser_tsplib
import utils  # Importa seu novo arquivo de utilidades

# --- Carregar os dados (como antes) ---
caminho_st70 = "data/st70.tsp"
minhas_cidades = parser_tsplib.carregar_cidades(caminho_st70)
num_cidades = len(minhas_cidades)

print(f"Arquivo: {caminho_st70}")
print(f"Total de cidades carregadas: {num_cidades}")
print("---------------------------------")


# --- Teste 1: Distância entre as duas primeiras cidades ---
cidade_0 = minhas_cidades[0] # (64.0, 96.0)
cidade_1 = minhas_cidades[1] # (80.0, 39.0)
dist = utils.calcular_distancia(cidade_0, cidade_1)
print(f"Distância entre cidade 0 {cidade_0} e cidade 1 {cidade_1}: {dist:.2f}")


# --- Teste 2: Custo de uma rota aleatória ---
# 1. Cria uma rota aleatória
rota_teste = utils.criar_rota_aleatoria(num_cidades)

# 2. Calcula o custo total dessa rota
custo_rota = utils.calcular_distancia_total(rota_teste, minhas_cidades)

print(f"\nCusto de uma rota aleatória com {num_cidades} cidades: {custo_rota:.2f}")
print("Rota aleatória (primeiras 10 cidades):")
print(rota_teste[:10])
print("---------------------------------")