"""
main.py - PROJETO FINAL COMPLETO
Estruturado para permitir execução seletiva das fases de análise.
"""
import os
import time
import numpy as np
from datetime import datetime
from scipy import stats

# Módulos do Projeto
import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado

# ==============================================================================
# 1. CONTROLE DE EXECUÇÃO (O "SELETOR")
# ==============================================================================
MODO_TESTE = False  # False = 30 execuções (Entrega) | True = 2 execuções (Teste)

# LISTA DE FASES ATIVAS
# Para a entrega final, o código mostra que tudo foi implementado.
FASES_PARA_RODAR = [
    # --- JÁ EXECUTADOS  ---
    # "FASE_1_POPULACAO",
    # "FASE_2_SELECAO",
    # "FASE_3_CROSSOVER",
    # "FASE_4_TIPO_MUTACAO",
    # "FASE_FINAL",

    # --- O QUE FALTA  ---
    "FASE_3B_TAXA_MUTACAO",  # Análise 3: Taxa 1% vs 5%
    "FASE_4B_ELITISMO",      # Análise 4: Sem Elitismo vs 5% vs 10%
    "FASE_6_CALIBRACAO_ACO", # Análise 6/7: Ajuste do Novo Algoritmo
]

if MODO_TESTE:
    print("\n⚠️  ALERTA: MODO DE TESTE (Rápido)")
    NUM_EXECUCOES = 2
    NUM_GERACOES = 10
    TAM_POP_PADRAO = 10
else:
    print("\n🚀 MODO DE ENTREGA (Estatístico - 30 execs)")
    NUM_EXECUCOES = 30
    NUM_GERACOES = 500
    TAM_POP_PADRAO = 50

INSTANCIAS = ["data/st70.tsp", "data/eil101.tsp", "data/ch130.tsp"]

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================
def garantir_pasta(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)

def barra_progresso(atual, total, texto):
    percent = (atual / total) * 100
    bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
    print(f"\r|{bar}| {percent:.1f}% - {texto}", end="", flush=True)

def rodar_bateria(titulo_fase, pasta_fase, lista_experimentos):
    print(f"\n\n{'='*60}")
    print(f"📂 INICIANDO: {titulo_fase}")
    print(f"{'='*60}")
    
    garantir_pasta(f"resultados/{pasta_fase}")
    sumario_global = {}
    resultados_raw_global = {}

    total_passos = len(INSTANCIAS) * len(lista_experimentos) * NUM_EXECUCOES
    passo_atual = 0

    for arquivo_tsp in INSTANCIAS:
        nome_instancia = os.path.basename(arquivo_tsp).split(".")[0]
        pasta_saida = f"resultados/{pasta_fase}/{nome_instancia}"
        garantir_pasta(pasta_saida)
        
        try:
            cidades = parser_tsplib.carregar_cidades(arquivo_tsp)
            matriz_dist = utils.calcular_matriz_distancias(cidades)
        except Exception as e:
            print(f"Erro crítico ao ler {arquivo_tsp}: {e}")
            continue

        melhor_dist_instancia = float("inf")
        melhor_alg_instancia = "N/A"
        melhor_rota_objeto = None
        
        dados_boxplot = {}
        dados_tempos = {}
        imagens_convergencia = []

        for exp in lista_experimentos:
            nome_exp = exp["nome"]
            tipo_alg = exp["algoritmo"]
            params = exp["params"]
            
            pop_size = params.get("tamanho_populacao", TAM_POP_PADRAO)
            params_limpo = params.copy()
            if "tamanho_populacao" in params_limpo: del params_limpo["tamanho_populacao"]

            lista_distancias = []
            lista_tempos = []
            historico_melhores = []
            historico_medias = []

            for i in range(NUM_EXECUCOES):
                passo_atual += 1
                barra_progresso(passo_atual, total_passos, f"{nome_instancia} > {nome_exp}")
                
                inicio = time.perf_counter()
                
                if tipo_alg == "AG":
                    modelo = AlgoritmoGenetico(cidades, dist_matrix=matriz_dist, **params_limpo, 
                                             num_geracoes=NUM_GERACOES, tamanho_populacao=pop_size)
                    resultado = modelo.executar()
                    historico_medias.append(modelo.historico_medias)
                    historico_melhores.append(modelo.historico_melhores)
                    dist_final = resultado.distancia
                    rota_final = resultado.rota

                elif tipo_alg == "ACO":
                    modelo = ACO(cidades, **params_limpo, num_iteracoes=NUM_GERACOES)
                    rota_final, dist_final = modelo.executar()
                    historico_melhores.append(modelo.historico_melhores)
                
                elif tipo_alg == "SA":
                    modelo = RecozimentoSimulado(cidades, dist_matrix=matriz_dist, **params_limpo, 
                                               max_iterations=NUM_GERACOES*pop_size)
                    rota_final, dist_final = modelo.executar()
                    historico_melhores.append(modelo.historico_melhores)

                tempo_gasto = time.perf_counter() - inicio
                lista_distancias.append(dist_final)
                lista_tempos.append(tempo_gasto)

                if dist_final < melhor_dist_instancia:
                    melhor_dist_instancia = dist_final
                    melhor_alg_instancia = nome_exp
                    melhor_rota_objeto = rota_final

            dados_boxplot[nome_exp] = lista_distancias
            dados_tempos[nome_exp] = np.mean(lista_tempos)
            
            arquivo_conv = f"convergencia_{nome_exp}.png"
            visualizacao.plotar_convergencia(historico_melhores, historico_medias if tipo_alg=="AG" else None, 
                                           f"{pasta_saida}/{arquivo_conv}")
            imagens_convergencia.append({"nome": nome_exp, "arquivo": arquivo_conv})

        # Relatórios
        visualizacao.plotar_boxplot_comparativo(dados_boxplot, f"{pasta_saida}/boxplot_{nome_instancia}.png")
        visualizacao.plotar_comparativo_tempo(dados_tempos, f"{pasta_saida}/tempos_{nome_instancia}.png")
        if melhor_rota_objeto:
            visualizacao.plotar_rota(melhor_rota_objeto, cidades, f"{pasta_saida}/melhor_rota_{nome_instancia}.png")

        html_stats = ""
        chaves = list(dados_boxplot.keys())
        for i in range(len(chaves)):
            for j in range(i+1, len(chaves)):
                try:
                    s, p = stats.ttest_ind(dados_boxplot[chaves[i]], dados_boxplot[chaves[j]], equal_var=False)
                    cor = "var(--success)" if p < 0.05 else "var(--text-muted)"
                    html_stats += f"<p><b>{chaves[i]} vs {chaves[j]}</b>: p={p:.4f} <span style='color:{cor}'>({'Diferença Real' if p<0.05 else 'Empate'})</span></p>"
                except: pass

        imagens = {"rota": f"melhor_rota_{nome_instancia}.png", "tempos": f"tempos_{nome_instancia}.png", "boxplot": f"boxplot_{nome_instancia}.png", "convergencias": imagens_convergencia}
        visualizacao.gerar_relatorio_instancia(f"{pasta_saida}/relatorio_{nome_instancia}.html", nome_instancia, titulo_fase, melhor_alg_instancia, melhor_dist_instancia, html_stats, imagens)
        
        sumario_global[nome_instancia] = {"melhor_dist": melhor_dist_instancia, "melhor_alg": melhor_alg_instancia}
        for k, v in dados_boxplot.items(): resultados_raw_global[f"{nome_instancia}_{k}"] = v

    visualizacao.plotar_boxplot_comparativo(resultados_raw_global, f"resultados/{pasta_fase}/boxplot_global.png")
    lista_vencedores = [v['melhor_alg'] for v in sumario_global.values()]
    kpis = {"melhor_alg": max(set(lista_vencedores), key=lista_vencedores.count) if lista_vencedores else "N/A", "melhor_dist": min([v['melhor_dist'] for v in sumario_global.values()]) if sumario_global else 0.0, "total_exec": total_passos}
    visualizacao.gerar_relatorio_final(f"resultados/{pasta_fase}/index.html", sumario_global, kpis, titulo_fase)
    print(f"\n✅ FASE CONCLUÍDA: {titulo_fase}")

# ==============================================================================
# 3. EXECUÇÃO SELETIVA
# ==============================================================================
print("\n>>> INICIANDO SISTEMA DE BENCHMARKING (SELETIVO) <<<")

if "FASE_1_POPULACAO" in FASES_PARA_RODAR:
    rodar_bateria("Fase 1: Calibração (População)", "1_calibracao_parametros", [
        {"nome": "AG_Pop50",  "algoritmo": "AG", "params": {"tamanho_populacao": 50,  "taxa_mutacao": 0.05}},
        {"nome": "AG_Pop100", "algoritmo": "AG", "params": {"tamanho_populacao": 100, "taxa_mutacao": 0.05}}
    ])

if "FASE_2_SELECAO" in FASES_PARA_RODAR:
    rodar_bateria("Fase 2: Comparativo de Seleção", "2_analise_selecao", [
        {"nome": "AG_Torneio", "algoritmo": "AG", "params": {"metodo_selecao": "torneio", "taxa_mutacao": 0.05}},
        {"nome": "AG_Roleta",  "algoritmo": "AG", "params": {"metodo_selecao": "roleta",  "taxa_mutacao": 0.05}}
    ])

if "FASE_3_CROSSOVER" in FASES_PARA_RODAR:
    rodar_bateria("Fase 3: Comparativo de Crossover", "3_analise_crossover", [
        {"nome": "AG_OX",  "algoritmo": "AG", "params": {"metodo_crossover": "ox",  "metodo_selecao": "torneio", "taxa_mutacao": 0.05}},
        {"nome": "AG_PMX", "algoritmo": "AG", "params": {"metodo_crossover": "pmx", "metodo_selecao": "torneio", "taxa_mutacao": 0.05}}
    ])

if "FASE_3B_TAXA_MUTACAO" in FASES_PARA_RODAR:
    # REQUISITO NOVO: Analisar taxa de 1% vs 5%
    rodar_bateria("Fase 3B: Taxa de Mutação (1% vs 5%)", "3b_analise_taxa_mutacao", [
        {"nome": "AG_Mut1%", "algoritmo": "AG", "params": {"taxa_mutacao": 0.01, "metodo_mutacao": "inversion", "taxa_elitismo": 0.05}},
        {"nome": "AG_Mut5%", "algoritmo": "AG", "params": {"taxa_mutacao": 0.05, "metodo_mutacao": "inversion", "taxa_elitismo": 0.05}}
    ])

if "FASE_4_TIPO_MUTACAO" in FASES_PARA_RODAR:
    rodar_bateria("Fase 4: Tipo de Mutação", "4_analise_mutacao", [
        {"nome": "AG_Swap",      "algoritmo": "AG", "params": {"metodo_mutacao": "swap",      "metodo_crossover": "ox", "taxa_mutacao": 0.05}},
        {"nome": "AG_Inversion", "algoritmo": "AG", "params": {"metodo_mutacao": "inversion", "metodo_crossover": "ox", "taxa_mutacao": 0.05}}
    ])

if "FASE_4B_ELITISMO" in FASES_PARA_RODAR:
    # REQUISITO NOVO: Analisar impacto do elitismo
    rodar_bateria("Fase 4B: Comparativo de Elitismo", "4b_analise_elitismo", [
        {"nome": "AG_SemElitismo", "algoritmo": "AG", "params": {"taxa_elitismo": 0.0, "taxa_mutacao": 0.05}},
        {"nome": "AG_Elitismo5%",  "algoritmo": "AG", "params": {"taxa_elitismo": 0.05, "taxa_mutacao": 0.05}},
        {"nome": "AG_Elitismo10%", "algoritmo": "AG", "params": {"taxa_elitismo": 0.10, "taxa_mutacao": 0.05}}
    ])

if "FASE_6_CALIBRACAO_ACO" in FASES_PARA_RODAR:
    # REQUISITO NOVO: Calibrar o "Novo Algoritmo" (ACO) antes do final
    rodar_bateria("Fase 6: Calibração ACO (Rho)", "6_calibracao_aco", [
        {"nome": "ACO_Rho0.1", "algoritmo": "ACO", "params": {"rho": 0.1, "alfa": 1.0, "beta": 2.5, "num_formigas": 50}},
        {"nome": "ACO_Rho0.5", "algoritmo": "ACO", "params": {"rho": 0.5, "alfa": 1.0, "beta": 2.5, "num_formigas": 50}}
    ])

if "FASE_FINAL" in FASES_PARA_RODAR:
    rodar_bateria("Fase Final: Comparativo Meta-heurísticas", "5_comparativo_final", [
        {"nome": "AG_Final",  "algoritmo": "AG",  "params": {"taxa_mutacao": 0.05, "metodo_crossover": "ox", "metodo_mutacao": "inversion", "taxa_elitismo": 0.05}},
        {"nome": "ACO_Final", "algoritmo": "ACO", "params": {"num_formigas": 50, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        {"nome": "SA_Final",  "algoritmo": "SA",  "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
    ])

print("\n🎉 PROCESSAMENTO CONCLUÍDO! Verifique a pasta 'resultados/'.")