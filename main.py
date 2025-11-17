import parser_tsplib
import utils
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO # NOVO: Importa a classe ACO

# --- 0. ESCOLHA DO ALGORITMO ---
ALGORITMO_A_EXECUTAR = "ACO" # Mude para "ACO" para testar
# ---------------------------------

# --- 1. CONFIGURAÇÕES GERAIS ---
ARQUIVO_DADOS = "data/st70.tsp"
TAMANHO_POPULACAO = 50  # Usado pelo AG e ACO (num_formigas)
NUM_GERACOES = 200      # Usado pelo AG e ACO (num_iteracoes)
TAXA_MUTACAO = 0.01     # Apenas AG
TAXA_ELITISMO = 0.05    # Apenas AG

# --- PARÂMETROS PARA AS ANÁLISES DO AG ---
METODO_SELECAO = 'torneio' 
METODO_MUTACAO = 'inversao' 
METODO_CROSSOVER = 'ox'

# --- PARÂMETROS PARA AS ANÁLISES DO ACO ---
ACO_ALFA = 1.0 # Peso do feromônio
ACO_BETA = 5.0 # Peso da distância
ACO_RHO = 0.1  # Taxa de evaporação
ACO_Q = 100    # Quantidade de feromônio

# -----------------------------------------------------------------

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


if ALGORITMO_A_EXECUTAR == "AG":
    # --- 3. INICIALIZAÇÃO DO ALGORITMO GENÉTICO ---
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
    print("\n--- RESULTADO FINAL (AG) ---")
    print(f"Melhor solução encontrada após {NUM_GERACOES} gerações:")
    print(f"Distância: {melhor_solucao.distancia:.2f}")
    print(f"Rota (primeiras 10 cidades): {melhor_solucao.rota[:10]}...")
    print("---------------------------------")

elif ALGORITMO_A_EXECUTAR == "ACO":
    # --- 3. INICIALIZAÇÃO DA COLÔNIA DE FORMIGAS ---
    aco = ACO(
        cidades=minhas_cidades,
        num_formigas=TAMANHO_POPULACAO,
        num_iteracoes=NUM_GERACOES,
        alfa=ACO_ALFA,
        beta=ACO_BETA,
        rho=ACO_RHO,
        Q=ACO_Q
    )
    
    # --- 4. EXECUÇÃO DO ACO ---
    # Chama o loop principal
    melhor_rota, melhor_distancia = aco.executar()
    
    # --- 5. RESULTADO FINAL ---
    print("\n--- RESULTADO FINAL (ACO) ---")
    print(f"Melhor solução encontrada após {NUM_GERACOES} iterações:")
    print(f"Distância: {melhor_distancia:.2f}")
    print(f"Rota (primeiras 10 cidades): {melhor_rota[:10]}...")
    print("---------------------------------")
    
    # Próximo passo:
    # melhor_solucao = aco.executar()