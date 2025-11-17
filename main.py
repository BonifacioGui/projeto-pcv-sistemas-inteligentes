import parser_tsplib
import utils
from algoritmo_genetico import AlgoritmoGenetico, Individuo 

# --- 1. CONFIGURAÇÕES GERAIS ---
#    (Aqui você irá alterar os valores para rodar os EXPERIMENTOS)
# -----------------------------------------------------------------
ARQUIVO_DADOS = "data/st70.tsp"
TAMANHO_POPULACAO = 50
NUM_GERACOES = 200 
TAXA_MUTACAO = 0.01 # 1%

# --- PARÂMETROS PARA AS ANÁLISES ---
# Análise 2: 'torneio' ou 'roleta'
METODO_SELECAO = 'torneio' 

# Análise 3: 'troca' ou 'inversao'
METODO_MUTACAO = 'inversao' 

# Análise 4: 0.0 (Somente Filhos), 0.05 (5%), 0.1 (10%)
TAXA_ELITISMO = 0.05 

# (Futuro) Análise de Crossover: 'ox' ou 'pmx'
METODO_CROSSOVER = 'ox'

# -----------------------------------------------------------------

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


# --- 3. INICIALIZAÇÃO DO ALGORITMO GENÉTICO ---
# ATUALIZADO: Passa as configurações para o construtor
ag = AlgoritmoGenetico(
    cidades=minhas_cidades,
    tamanho_populacao=TAMANHO_POPULACAO,
    num_geracoes=NUM_GERACOES,
    taxa_mutacao=TAXA_MUTACAO,
    taxa_elitismo=TAXA_ELITISMO,
    metodo_selecao=METODO_SELECAO,
    metodo_mutacao=METODO_MUTACAO,
    metodo_crossover=METODO_CROSSOVER
)

# --- 4. EXECUÇÃO DA EVOLUÇÃO ---
melhor_solucao = ag.executar() # Chama o loop principal

# --- 5. RESULTADO FINAL ---
print("\n--- RESULTADO FINAL ---")
print(f"Melhor solução encontrada após {NUM_GERACOES} gerações:")
print(f"Distância: {melhor_solucao.distancia:.2f}")
print(f"Rota (primeiras 10 cidades): {melhor_solucao.rota[:10]}...")
print("---------------------------------")