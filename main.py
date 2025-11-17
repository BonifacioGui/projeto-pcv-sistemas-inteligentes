import parser_tsplib
import utils
import time 
import numpy as np 
import visualizacao # Importa nosso arquivo de gráficos
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO 
from recozimento_simulado import RecozimentoSimulado 

# --- 1. DEFINIÇÃO DOS EXPERIMENTOS ---
NUM_EXECUCOES = 30
ARQUIVO_DADOS = "data/st70.tsp"
NUM_GERACOES = 200      # (AG, ACO)
TAMANHO_POPULACAO = 50  # (AG, ACO)

experimentos_para_rodar = [
    {
        "nome": "AG-Torneio-Troca",
        "algoritmo": "AG",
        "params": {
            "metodo_selecao": "torneio",
            "metodo_mutacao": "troca",
            "taxa_elitismo": 0.05,
            "taxa_mutacao": 0.01,
            "metodo_crossover": "ox"
        }
    },
    {
        "nome": "AG-Torneio-Inversao",
        "algoritmo": "AG",
        "params": {
            "metodo_selecao": "torneio",
            "metodo_mutacao": "inversao",
            "taxa_elitismo": 0.05,
            "taxa_mutacao": 0.01,
            "metodo_crossover": "ox"
        }
    }
]

# -----------------------------------------------------------------

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando dados do problema...")
minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
num_cidades = len(minhas_cidades)
print(f"Arquivo: {ARQUIVO_DADOS} ({num_cidades} cidades)")
print("---------------------------------")


# --- 3. LOOP DE EXECUÇÃO DOS EXPERIMENTOS ---
resultados_para_boxplot = {}
resultados_para_convergencia = {}

# NOVO: Variáveis para guardar a MELHOR ROTA DE TODAS
melhor_rota_geral = None
melhor_distancia_geral = float('inf')

for exp in experimentos_para_rodar:
    nome_exp = exp["nome"]
    algoritmo = exp["algoritmo"]
    
    print(f"\n===== INICIANDO EXPERIMENTO: {nome_exp} ({NUM_EXECUCOES} execuções) =====")
    
    resultados_distancia = []
    resultados_tempo = []
    historico_30_execucoes = []
    
    for i in range(NUM_EXECUCOES):
        print(f"--- Execução {i + 1} de {NUM_EXECUCOES} ---", end="") 
        start_time = time.time()
        
        distancia_final = float('inf')
        rota_final = []
        
        if algoritmo == "AG":
            params = exp["params"]
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
            melhor_solucao = ag.executar() 
            distancia_final = melhor_solucao.distancia
            rota_final = melhor_solucao.rota # Salva a rota
            historico_30_execucoes.append(ag.historico_melhores)

        elif algoritmo == "ACO":
            aco = ACO(
                cidades=minhas_cidades,
                num_formigas=TAMANHO_POPULACAO,
                num_iteracoes=NUM_GERACOES,
                alfa=1.0, beta=5.0, rho=0.1, Q=100
            )
            rota_final, distancia_final = aco.executar() # Salva a rota
            historico_30_execucoes.append(aco.historico_melhores)

        elif algoritmo == "SA":
            sa = RecozimentoSimulado(
                cidades=minhas_cidades,
                temp_inicial=10000, temp_final=1, taxa_resfriamento=0.995
            )
            rota_final, distancia_final = sa.executar() # Salva a rota
            historico_30_execucoes.append(sa.historico_melhores)
        
        end_time = time.time()
        tempo_execucao = end_time - start_time
        
        resultados_distancia.append(distancia_final)
        resultados_tempo.append(tempo_execucao)
        
        # NOVO: Verifica se esta é a melhor rota GERAL
        if distancia_final < melhor_distancia_geral:
            melhor_distancia_geral = distancia_final
            melhor_rota_geral = rota_final
        
        print(f" -> Distância = {distancia_final:.2f} (Tempo: {tempo_execucao:.2f}s)")

    # --- 4. RESUMO ESTATÍSTICO DO EXPERIMENTO ---
    print(f"\n--- Resumo Estatístico para: {nome_exp} ---")
    print(f"  Distância Média:   {np.mean(resultados_distancia):.2f}")
    print(f"  Distância Desv.Padrão: {np.std(resultados_distancia):.2f}")
    print(f"  Distância Melhor (Min):  {np.min(resultados_distancia):.2f}")
    print(f"  Tempo Médio:   {np.mean(resultados_tempo):.2f}s")
    
    # Salva os resultados para os gráficos finais
    resultados_para_boxplot[nome_exp] = resultados_distancia
    resultados_para_convergencia[nome_exp] = historico_30_execucoes

# --- 5. GERAÇÃO DE GRÁFICOS FINAIS ---
print("\n===== GERANDO GRÁFICOS COMPARATIVOS =====")

# 5.1. Gerar Gráfico de Convergência para cada experimento
for nome_exp, historico in resultados_para_convergencia.items():
    nome_grafico = f"convergencia_{nome_exp}.png"
    visualizacao.plotar_convergencia(historico, nome_grafico)

# 5.2. Gerar UM Boxplot comparando TODOS os experimentos
nome_boxplot = "boxplot_comparativo.png"
visualizacao.plotar_boxplot_comparativo(resultados_para_boxplot, nome_boxplot)

# 5.3. NOVO: Gerar UM gráfico da MELHOR ROTA encontrada
if melhor_rota_geral:
    visualizacao.plotar_rota(melhor_rota_geral, minhas_cidades, "melhor_rota_geral.png")
else:
    print("Nenhuma rota foi gerada para plotar.")

print("\nTodos os experimentos e gráficos foram concluídos.")