# main.py - Versão Final Integrada
# Requisitos atendidos:
# - 3 Algoritmos (AG, ACO, SA) [cite: 8, 21]
# - Estudo Paramétrico (AG Padrão vs AG Alta Mutação) 
# - Robustez (30 execuções) [cite: 46]
# - Gráficos de Convergência (Melhor e Média) [cite: 34]
# - Teste Estatístico T-Student 

import time
import json
import numpy as np
from scipy import stats # Necessário para o teste T-Student

# Importações dos módulos do projeto
import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado

# ======================================================================
# 1. CONFIGURAÇÕES GERAIS
# ======================================================================

NUM_EXECUCOES = 30
ARQUIVO_DADOS = "data/eil101.tsp" # Certifique-se que este arquivo existe
NUM_GERACOES = 500      # Critério de parada [cite: 20]
TAMANHO_POPULACAO = 50

# Lista de experimentos para o Estudo Paramétrico 
experimentos_para_rodar = [
    {
        "nome": "AG_Padrao",
        "algoritmo": "AG",
        "params": {
            "taxa_mutacao": 0.01, 
            "metodo_crossover": "ox",
            "metodo_mutacao": "swap"
        }
    },
    {
        "nome": "AG_AltaMutacao", # Variação de parâmetro para análise
        "algoritmo": "AG",
        "params": {
            "taxa_mutacao": 0.10, # 10% de mutação (vs 1% do padrão)
            "metodo_crossover": "ox",
            "metodo_mutacao": "swap"
        }
    },
    {
        "nome": "ACO_Padrao",
        "algoritmo": "ACO",
        "params": {
            "num_formigas": 20,
            "alfa": 1.0,
            "beta": 2.5,
            "rho": 0.1
        }
    },
    {
        "nome": "SA_Padrao",
        "algoritmo": "SA",
        "params": {
            "temp_inicial": 1000,
            "taxa_resfriamento": 0.995
        }
    }
]

# ======================================================================
# 2. CARREGAMENTO DOS DADOS
# ======================================================================
print(f"Carregando dados: {ARQUIVO_DADOS}...")
try:
    minhas_cidades = parser_tsplib.carregar_cidades(ARQUIVO_DADOS)
    # Pré-calcula matriz de distâncias (Otimização de Desempenho) [cite: 41]
    dist_matrix = utils.calcular_matriz_distancias(minhas_cidades)
    num_cidades = len(minhas_cidades)
    print(f"Carregado com sucesso: {num_cidades} cidades.")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o arquivo {ARQUIVO_DADOS}.")
    print(f"Detalhe: {e}")
    exit()

# ======================================================================
# 3. EXECUÇÃO DOS EXPERIMENTOS
# ======================================================================

# Dicionários globais para gráficos finais
resultados_para_boxplot = {}
resultados_para_convergencia_melhor = {} # Para plotar depois se quiser tudo junto
coleta_final_distancias = {} # Para o teste T-Student

melhor_rota_geral = None
melhor_distancia_geral = float("inf")

for exp in experimentos_para_rodar:
    nome_exp = exp["nome"]
    algoritmo = exp["algoritmo"]
    params = exp.get("params", {})

    print(f"\n===== EXPERIMENTO: {nome_exp} ({algoritmo}) =====")

    # Listas para armazenar as 30 execuções deste experimento
    resultados_distancia = []
    resultados_tempo = []
    
    historico_30_melhores = [] # Guarda a evolução do MELHOR
    historico_30_medias = []   # Guarda a evolução da MÉDIA (só AG)

    for exe in range(NUM_EXECUCOES):
        # Feedback detalhado: Mostra qual execução está rodando
        print(f"  > [{nome_exp}] Execução {exe + 1}/{NUM_EXECUCOES}...", end="", flush=True)
        
        start_time = time.perf_counter()        
        start_time = time.perf_counter()
        distancia_final = float("inf")
        rota_final = []
        
        try:
            # --- INSTANCIAÇÃO E EXECUÇÃO ---
            if algoritmo == "AG":
                ag = AlgoritmoGenetico(
                    cidades=minhas_cidades,
                    tamanho_populacao=TAMANHO_POPULACAO,
                    num_geracoes=NUM_GERACOES,
                    taxa_mutacao=params.get("taxa_mutacao", 0.01),
                    taxa_elitismo=0.05,
                    metodo_selecao="torneio",
                    metodo_crossover=params.get("metodo_crossover", "ox"),
                    metodo_mutacao=params.get("metodo_mutacao", "swap"),
                    dist_matrix=dist_matrix
                )
                melhor = ag.executar()
                
                distancia_final = melhor.distancia
                rota_final = melhor.rota
                
                # Coleta históricos (Melhor e Média)
                historico_30_melhores.append(ag.historico_melhores)
                historico_30_medias.append(ag.historico_medias)

            elif algoritmo == "ACO":
                aco = ACO(
                    cidades=minhas_cidades,
                    num_formigas=params.get("num_formigas", 20),
                    num_iteracoes=NUM_GERACOES, # Usamos num_geracoes como iterações para ser justo
                    alfa=params.get("alfa", 1.0),
                    beta=params.get("beta", 2.5),
                    rho=params.get("rho", 0.1)
                )
                rota_final, distancia_final = aco.executar()
                
                historico_30_melhores.append(aco.historico_melhores)
                # ACO não tem "média da população" da mesma forma que AG, enviamos None depois

            elif algoritmo == "SA":
                sa = RecozimentoSimulado(
                    cidades=minhas_cidades,
                    temp_inicial=params.get("temp_inicial", 1000),
                    max_iterations=NUM_GERACOES * TAMANHO_POPULACAO, # Ajuste de esforço computacional
                    dist_matrix=dist_matrix
                )
                rota_final, distancia_final = sa.executar()
                
                historico_30_melhores.append(sa.historico_melhores)

        except Exception as e:
            print(f"\n[ERRO na execução {exe}: {e}]")
            distancia_final = float("nan")

        # --- COLETA DE TEMPO E MELHOR GLOBAL ---
        tempo_execucao = time.perf_counter() - start_time
        
        # Feedback de conclusão da linha
        print(f" OK! (Tempo: {tempo_execucao:.2f}s | Dist: {distancia_final:.2f})")

        resultados_distancia.append(distancia_final)

        if distancia_final < melhor_distancia_geral:
            melhor_distancia_geral = distancia_final
            melhor_rota_geral = rota_final

    print(f"\n-> Média Distância: {np.nanmean(resultados_distancia):.2f} | Tempo Médio: {np.nanmean(resultados_tempo):.4f}s")

    # --- ARMAZENAMENTO PARA GRÁFICOS E ESTATÍSTICA ---
    resultados_para_boxplot[nome_exp] = resultados_distancia
    coleta_final_distancias[nome_exp] = resultados_distancia # Guarda para o teste T

    # --- GERAÇÃO DO GRÁFICO DE CONVERGÊNCIA (POR EXPERIMENTO) ---
    # Aqui atendemos o requisito: "mostrar como a melhor solução e a média evoluem" [cite: 34]
    if algoritmo == "AG":
        visualizacao.plotar_convergencia(
            historico_30_melhores, 
            historico_30_medias, 
            f"convergencia_{nome_exp}.png"
        )
    else:
        visualizacao.plotar_convergencia(
            historico_30_melhores, 
            None, 
            f"convergencia_{nome_exp}.png"
        )

# ======================================================================
# 4. GERAÇÃO DE GRÁFICOS FINAIS
# ======================================================================

# Boxplot comparativo [cite: 35]
visualizacao.plotar_boxplot_comparativo(
    resultados_para_boxplot,
    "boxplot_comparativo.png"
)

# Visualização da Melhor Rota encontrada [cite: 40]
if melhor_rota_geral:
    visualizacao.plotar_rota(
        melhor_rota_geral,
        minhas_cidades,
        "melhor_rota_encontrada.png"
    )

# ======================================================================
# 5. ANÁLISE ESTATÍSTICA (TESTE T-STUDENT)
# ======================================================================
# Requisito: "Aplicar testes estatísticos (ex. t-student) para validar as escolhas" 

print("\n===== ANÁLISE ESTATÍSTICA (T-STUDENT) =====")
print("Comparando AG_Padrao vs AG_AltaMutacao para validar parâmetros...")

if "AG_Padrao" in coleta_final_distancias and "AG_AltaMutacao" in coleta_final_distancias:
    data_A = np.array(coleta_final_distancias["AG_Padrao"])
    data_B = np.array(coleta_final_distancias["AG_AltaMutacao"])
    
    # Limpeza de NaNs (caso alguma execução tenha falhado)
    data_A = data_A[np.isfinite(data_A)]
    data_B = data_B[np.isfinite(data_B)]

    t_stat, p_valor = stats.ttest_ind(data_A, data_B)
    
    print(f"Resultados:")
    print(f"  Média AG_Padrao: {np.mean(data_A):.2f}")
    print(f"  Média AG_AltaMutacao: {np.mean(data_B):.2f}")
    print(f"  Estatística T: {t_stat:.4f}")
    print(f"  P-Valor: {p_valor:.4e}")
    
    alpha = 0.05
    if p_valor < alpha:
        print(f"  -> CONCLUSÃO: Diferença SIGNIFICATIVA (p < 0.05).")
        if np.mean(data_A) < np.mean(data_B):
            print("  -> O 'AG_Padrao' foi estatisticamente superior.")
        else:
            print("  -> O 'AG_AltaMutacao' foi estatisticamente superior.")
    else:
        print("  -> CONCLUSÃO: Não há diferença estatística significativa entre os parâmetros.")
else:
    print("Não foi possível realizar o teste (experimentos não encontrados).")

print("\nFim da execução.")