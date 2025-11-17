# Dentro de main.py

import parser_tsplib
import utils
from algoritmo_genetico import AlgoritmoGenetico, Individuo # Importa nossas NOVAS classes

# --- 1. CONFIGURAÇÕES GERAIS ---
ARQUIVO_DADOS = "data/st70.tsp"
TAMANHO_POPULACAO = 50
NUM_GERACOES = 100
TAXA_MUTACAO = 0.01 # 1%

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


# --- 3. INICIALIZAÇÃO DO ALGORITMO GENÉTICO ---

# 3.1. Cria a instância principal do AG
ag = AlgoritmoGenetico(
    cidades=minhas_cidades,
    tamanho_populacao=TAMANHO_POPULACAO,
    num_geracoes=NUM_GERACOES,
    taxa_mutacao=TAXA_MUTACAO
)

# 3.2. Cria a população inicial (nosso primeiro passo)
ag._criar_populacao_inicial()


# --- 4. VALIDAÇÃO DA POPULAÇÃO INICIAL ---
print("\n--- VALIDAÇÃO DA POPULAÇÃO INICIAL ---")
print(f"Tamanho da população: {len(ag.populacao)}")

# Encontra o melhor indivíduo (menor distância) na população ALEATÓRIA
# Usamos 'min' na 'distancia', pois queremos a MENOR distância.
melhor_individuo_inicial = min(ag.populacao, key=lambda ind: ind.distancia)

print(f"Melhor distância na Geração 0 (Aleatória): {melhor_individuo_inicial.distancia:.2f}")
print(f"Melhor indivíduo: {melhor_individuo_inicial}")
print("---------------------------------")

# Próximos passos (a serem implementados):
# ag.executar_evolucao()
# ag.mostrar_resultados()