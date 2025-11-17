# --- 0. IMPORTAÇÕES ---
# Importa nossas bibliotecas e classes
import parser_tsplib
import utils
import time 
import numpy as np 
import visualizacao # Nosso arquivo de gráficos
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO 
from recozimento_simulado import RecozimentoSimulado 

# --- 1. DEFINIÇÃO DOS EXPERIMENTOS ---
#
#    Este é o seu "Painel de Controle".
#    Basta modificar a lista 'experimentos_para_rodar' para
#    executar qualquer uma das Análises do projeto.
#
# -----------------------------------------------------------------
NUM_EXECUCOES = 30           # Número de execuções para robustez (conforme Análise 4)
ARQUIVO_DADOS = "data/st70.tsp" # Base de dados a ser usada (st70, eil101, ch130)
NUM_GERACOES = 200             # Critério de parada (AG, ACO)
TAMANHO_POPULACAO = 50         # Tamanho da população (AG) / N° de formigas (ACO)

# --- Lista de Experimentos para Executar ---
# (Este é o único bloco que você precisa mudar para cada análise)

# --- Lista de Experimentos para Executar ---

# --- Lista de Experimentos para Executar ---

# Análise 4: Comparação de Taxas de Elitismo
experimentos_para_rodar = [
    {
        "nome": "AG-Elitismo-0%",
        "algoritmo": "AG",
        "params": {
            "metodo_selecao": "torneio",    # (Constante)
            "metodo_mutacao": "inversao",   # (Constante)
            "taxa_elitismo": 0.0,           # <-- VARIÁVEL DE TESTE
            "taxa_mutacao": 0.01,           # (Constante)
            "metodo_crossover": "ox"        # (Constante)
        }
    },
    {
        "nome": "AG-Elitismo-5%",
        "algoritmo": "AG",
        "params": {
            "metodo_selecao": "torneio",    # (Constante)
            "metodo_mutacao": "inversao",   # (Constante)
            "taxa_elitismo": 0.05,          # <-- VARIÁVEL DE TESTE
            "taxa_mutacao": 0.01,           # (Constante)
            "metodo_crossover": "ox"        # (Constante)
        }
    },
    {
        "nome": "AG-Elitismo-10%",
        "algoritmo": "AG",
        "params": {
            "metodo_selecao": "torneio",    # (Constante)
            "metodo_mutacao": "inversao",   # (Constante)
            "taxa_elitismo": 0.1,           # <-- VARIÁVEL DE TESTE
            "taxa_mutacao": 0.01,           # (Constante)
            "metodo_crossover": "ox"        # (Constante)
        }
    }
]
# -----------------------------------------------------------------

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
# Chama o parser para ler o arquivo e pegar a lista de (x,y)
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


# --- 3. LOOP DE EXECUÇÃO DOS EXPERIMENTOS ---
# Dicionários para guardar os resultados de TODOS os experimentos
resultados_para_boxplot = {}
resultados_para_convergencia = {}

# Guarda a melhor rota encontrada entre TODOS os experimentos
melhor_rota_geral = None
melhor_distancia_geral = float('inf') # Começa com infinito

# Loop principal: itera sobre a lista 'experimentos_para_rodar'
for exp in experimentos_para_rodar:
    nome_exp = exp["nome"]
    algoritmo = exp["algoritmo"]
    
    print(f"\n===== INICIANDO EXPERIMENTO: {nome_exp} ({NUM_EXECUCOES} execuções) =====")
    
    # Listas para guardar os resultados das 30 execuções DESTE experimento
    resultados_distancia = []
    resultados_tempo = []
    historico_30_execucoes = []
    
    # Loop interno: Roda o experimento 30 vezes
    for i in range(NUM_EXECUCOES):
        print(f"--- Execução {i + 1} de {NUM_EXECUCOES} ---", end="") # 'end=""' imprime na mesma linha
        start_time = time.time() # Marca o tempo de início
        
        distancia_final = float('inf')
        rota_final = []
        
        # --- Bloco do ALGORITMO GENÉTICO ---
        if algoritmo == "AG":
            params = exp["params"] # Pega os parâmetros do AG (ex: 'metodo_selecao')
            # Instancia o AG passando as configurações
            ag = AlgoritmoGenetico(
                cidades=minhas_cidades,
                tamanho_populacao=TAMANHO_POPULACAO,
                num_geracoes=NUM_GERACOES,
                taxa_mutacao=params["taxa_mutacao"],
                taxa_elitismo=params["taxa_elitismo"],
                metodo_selecao=params["metodo_selecao"],
                metodo_mutacao=params["metodo_mutacao"],
                metodo_crossover=params["metodo_crossover"]
            )
            # Roda a evolução
            melhor_solucao = ag.executar() 
            # Salva os resultados
            distancia_final = melhor_solucao.distancia
            rota_final = melhor_solucao.rota
            historico_30_execucoes.append(ag.historico_melhores)

        # --- Bloco da COLÔNIA DE FORMIGAS ---
        elif algoritmo == "ACO":
            aco = ACO(
                cidades=minhas_cidades,
                num_formigas=TAMANHO_POPULACAO,
                num_iteracoes=NUM_GERACOES,
                alfa=1.0, beta=5.0, rho=0.1, Q=100 # (Valores padrão)
            )
            # Roda a evolução
            rota_final, distancia_final = aco.executar()
            # Salva os resultados
            historico_30_execucoes.append(aco.historico_melhores)

        # --- Bloco do RECOZIMENTO SIMULADO ---
        elif algoritmo == "SA":
            sa = RecozimentoSimulado(
                cidades=minhas_cidades,
                temp_inicial=10000, temp_final=1, taxa_resfriamento=0.995 # (Valores padrão)
            )
            # Roda a evolução
            rota_final, distancia_final = sa.executar()
            # Salva os resultados
            historico_30_execucoes.append(sa.historico_melhores)
        
        # --- Coleta de dados da execução ---
        end_time = time.time()
        tempo_execucao = end_time - start_time
        
        # Salva a distância e o tempo desta execução
        resultados_distancia.append(distancia_final)
        resultados_tempo.append(tempo_execucao)
        
        # Verifica se esta é a MELHOR ROTA GERAL já vista
        if distancia_final < melhor_distancia_geral:
            melhor_distancia_geral = distancia_final
            melhor_rota_geral = rota_final
        
        print(f" -> Distância = {distancia_final:.2f} (Tempo: {tempo_execucao:.2f}s)")

    # --- 4. RESUMO ESTATÍSTICO DO EXPERIMENTO ---
    print(f"\n--- Resumo Estatístico para: {nome_exp} ---")
    print(f"  Distância Média:   {np.mean(resultados_distancia):.2f}")
    print(f"  Distância Desv.Padrão: {np.std(resultados_distancia):.2f} (Robustez)")
    print(f"  Distância Melhor (Min):  {np.min(resultados_distancia):.2f}")
    print(f"  Tempo Médio:   {np.mean(resultados_tempo):.2f}s")
    
    # Salva os resultados agregados para os gráficos finais
    resultados_para_boxplot[nome_exp] = resultados_distancia
    resultados_para_convergencia[nome_exp] = historico_30_execucoes

# --- 5. GERAÇÃO DE GRÁFICOS FINAIS ---
print("\n===== GERANDO GRÁFICOS COMPARATIVOS =====")

# NOVO: Limpa o nome do arquivo de dados para usar nos títulos dos gráficos
# "data/st70.tsp" -> "st70"
nome_base_arquivo = ARQUIVO_DADOS.replace("data/", "").replace(".tsp", "")

# 5.1. Gerar Gráfico de Convergência para cada experimento
for nome_exp, historico in resultados_para_convergencia.items():
    # Cria um nome de arquivo limpo, ex: "convergencia_st70_AG-Crossover-OX.png"
    nome_grafico = f"convergencia_{nome_base_arquivo}_{nome_exp}.png"
    visualizacao.plotar_convergencia(historico, nome_grafico)

# 5.2. Gerar UM Boxplot comparando TODOS os experimentos
# CORRIGIDO: Usa o nome_base_arquivo limpo
nome_boxplot = f"boxplot_{nome_base_arquivo}.png"
visualizacao.plotar_boxplot_comparativo(resultados_para_boxplot, nome_boxplot)

# 5.3. Gerar UM gráfico da MELHOR ROTA encontrada
if melhor_rota_geral:
    # CORRIGIDO: Usa o nome_base_arquivo limpo
    nome_rota = f"melhor_rota_{nome_base_arquivo}.png"
    visualizacao.plotar_rota(melhor_rota_geral, minhas_cidades, nome_rota)
else:
    print("Nenhuma rota foi gerada para plotar.")

print("\nTodos os experimentos e gráficos foram concluídos.")