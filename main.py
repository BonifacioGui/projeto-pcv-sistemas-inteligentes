import parser_tsplib
import utils
from algoritmo_genetico import AlgoritmoGenetico, Individuo 

# --- 1. CONFIGURAÇÕES GERAIS ---
ARQUIVO_DADOS = "data/st70.tsp"
TAMANHO_POPULACAO = 50
NUM_GERACOES = 200 # Aumentei para 200 para ver melhor a evolução
TAXA_MUTACAO = 0.01 # 1%

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


# --- 3. INICIALIZAÇÃO DO ALGORITMO GENÉTICO ---
ag = AlgoritmoGenetico(
    cidades=minhas_cidades,
    tamanho_populacao=TAMANHO_POPULACAO,
    num_geracoes=NUM_GERACOES,
    taxa_mutacao=TAXA_MUTACAO
)

# --- 4. EXECUÇÃO DA EVOLUÇÃO ---
print("\nIniciando evolução do AG...")
melhor_solucao = ag.executar() # Chama o loop principal

# --- 5. RESULTADO FINAL ---
print("\n--- RESULTADO FINAL ---")
print(f"Melhor solução encontrada após {NUM_GERACOES} gerações:")
print(f"Distância: {melhor_solucao.distancia:.2f}")
print(f"Rota (primeiras 10 cidades): {melhor_solucao.rota[:10]}...")
print("---------------------------------")