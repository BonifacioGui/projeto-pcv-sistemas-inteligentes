"""
main.py - PROJETO FINAL COMPLETO: CAIXEIRO VIAJANTE
Estruturado para cumprir todos os requisitos experimentais do Relatório Técnico.

HISTÓRICO DE EXECUÇÃO:
- Fases 1, 2, 3, 4A e 5: JÁ EXECUTADAS (Resultados preservados em 'resultados/')
- Fases 3B, 4B, 6 e 7: ATIVAS NESTA EXECUÇÃO (Complementos exigidos)
"""
import os
import time
import numpy as np
from scipy import stats

# Módulos do Projeto
import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado

# ==============================================================================
# 1. CONFIGURAÇÃO DE EXECUÇÃO
# ==============================================================================
# Mantenha False para a entrega final (30 execuções garantem a validade estatística)
MODO_TESTE = False 

if MODO_TESTE:
    print("\n⚠️  ALERTA: MODO DE TESTE (Rápido - Resultados não valem nota)")
    NUM_EXECUCOES = 2
    NUM_GERACOES = 10
    TAM_POP_PADRAO = 10
else:
    print("\n🚀 MODO DE ENTREGA (30 Execuções - Estatística Robusta)")
    NUM_EXECUCOES = 30
    NUM_GERACOES = 500
    TAM_POP_PADRAO = 50

INSTANCIAS = ["data/st70.tsp", "data/eil101.tsp", "data/ch130.tsp"]

# ==============================================================================
# 2. FUNÇÕES DO SISTEMA (Motor de Execução)
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
        dados_boxplot = {}
        dados_tempos = {}
        historico_melhores_plot = []

        for exp in lista_experimentos:
            nome_exp = exp["nome"]
            tipo_alg = exp["algoritmo"]
            params = exp["params"].copy()
            pop_size = params.pop("tamanho_populacao", TAM_POP_PADRAO)

            lista_distancias = []
            lista_tempos = []
            historico_melhores = []
            
            # --- LOOP DE EXECUÇÕES (30x) ---
            for i in range(NUM_EXECUCOES):
                passo_atual += 1
                barra_progresso(passo_atual, total_passos, f"{nome_instancia} > {nome_exp}")
                
                inicio = time.perf_counter()
                
                if tipo_alg == "AG":
                    modelo = AlgoritmoGenetico(cidades, dist_matrix=matriz_dist, **params, 
                                             num_geracoes=NUM_GERACOES, tamanho_populacao=pop_size)
                    res = modelo.executar()
                    dist_final = res.distancia
                    historico_melhores.append(modelo.historico_melhores)

                elif tipo_alg == "ACO":
                    modelo = ACO(cidades, **params, num_iteracoes=NUM_GERACOES)
                    _, dist_final = modelo.executar()
                    historico_melhores.append(modelo.historico_melhores)
                
                elif tipo_alg == "SA":
                    modelo = RecozimentoSimulado(cidades, dist_matrix=matriz_dist, **params, 
                                               max_iterations=NUM_GERACOES*pop_size)
                    _, dist_final = modelo.executar()

                lista_tempos.append(time.perf_counter() - inicio)
                lista_distancias.append(dist_final)
                
                if dist_final < melhor_dist_instancia:
                    melhor_dist_instancia = dist_final
                    melhor_alg_instancia = nome_exp

            dados_boxplot[nome_exp] = lista_distancias
            dados_tempos[nome_exp] = np.mean(lista_tempos)
            
            # Gera gráfico de convergência para este experimento
            visualizacao.plotar_convergencia(historico_melhores, None, f"{pasta_saida}/convergencia_{nome_exp}.png")

        # --- GERAÇÃO DE RELATÓRIOS DA INSTÂNCIA ---
        visualizacao.plotar_boxplot_comparativo(dados_boxplot, f"{pasta_saida}/boxplot_{nome_instancia}.png")
        
        # Teste T Simplificado (apenas se houver 2 experimentos para comparar)
        html_stats = ""
        chaves = list(dados_boxplot.keys())
        if len(chaves) == 2:
            try:
                s, p = stats.ttest_ind(dados_boxplot[chaves[0]], dados_boxplot[chaves[1]], equal_var=False)
                vencedor = chaves[0] if np.mean(dados_boxplot[chaves[0]]) < np.mean(dados_boxplot[chaves[1]]) else chaves[1]
                html_stats += f"<p><b>Comparativo Estatístico:</b> {chaves[0]} vs {chaves[1]}<br>p-value: {p:.4f}<br>Vencedor Estatístico: {vencedor}</p>"
            except: pass

        # HTML simples para não quebrar se faltar imagem
        visualizacao.gerar_relatorio_instancia(f"{pasta_saida}/relatorio.html", nome_instancia, titulo_fase, melhor_alg_instancia, melhor_dist_instancia, html_stats, {"rota": "", "tempos": "", "boxplot": f"boxplot_{nome_instancia}.png", "convergencias": []})
        
        sumario_global[nome_instancia] = {"melhor": melhor_alg_instancia}
        for k, v in dados_boxplot.items(): resultados_raw_global[f"{nome_instancia}_{k}"] = v

    # Boxplot Global (Todas as instâncias lado a lado)
    visualizacao.plotar_boxplot_comparativo(resultados_raw_global, f"resultados/{pasta_fase}/boxplot_global.png")
    print(f"\n✅ FASE CONCLUÍDA: {titulo_fase} (Verifique a pasta 'resultados/{pasta_fase}')")

# ==============================================================================
# 3. LISTA DE EXPERIMENTOS (Roteiro do Relatório Técnico)
# ==============================================================================
print("\n>>> EXECUÇÃO DE EXPERIMENTOS COMPLEMENTARES <<<")

# -------------------------------------------------------------------------
# FASES JÁ EXECUTADAS (Mantidas como registro do trabalho realizado)
# -------------------------------------------------------------------------
# rodar_bateria("Fase 1: Calibração (População)", "1_calibracao_parametros", [...])
# rodar_bateria("Fase 2: Comparativo de Seleção", "2_analise_selecao", [...])
# rodar_bateria("Fase 3A: Tipo de Mutação", "4_analise_mutacao", [...]) # Nota: Pasta antiga nomeada como '4', mantido p/ histórico
# rodar_bateria("Fase 5: Comparativo Final", "5_comparativo_final", [...])

# -------------------------------------------------------------------------
# FASES COMPLEMENTARES (RODANDO AGORA)
# -------------------------------------------------------------------------

# --- ANÁLISE 3B: TAXA DE MUTAÇÃO ---
# Requisito PDF: "TAXA DE MUTAÇÃO 1% vs. 5%"
rodar_bateria("Fase 3B: Taxa de Mutação (1% vs 5%)", "3b_analise_taxa_mutacao", [
    {"nome": "AG_Mut_1%", "algoritmo": "AG", "params": {"taxa_mutacao": 0.01, "metodo_mutacao": "inversion"}},
    {"nome": "AG_Mut_5%", "algoritmo": "AG", "params": {"taxa_mutacao": 0.05, "metodo_mutacao": "inversion"}}
])

# --- ANÁLISE 4B: ELITISMO ---
# Requisito PDF: "SOMENTE FILHOS, ELITISMO 5%, ELITISMO 10%"
rodar_bateria("Fase 4B: Impacto do Elitismo", "4b_analise_elitismo", [
    {"nome": "AG_Sem_Elitismo", "algoritmo": "AG", "params": {"taxa_elitismo": 0.0,  "taxa_mutacao": 0.05}},
    {"nome": "AG_Elitismo_5%",  "algoritmo": "AG", "params": {"taxa_elitismo": 0.05, "taxa_mutacao": 0.05}},
    {"nome": "AG_Elitismo_10%", "algoritmo": "AG", "params": {"taxa_elitismo": 0.10, "taxa_mutacao": 0.05}}
])

# --- ANÁLISE 6: CALIBRAGEM ACO (Parâmetro 1) ---
# Requisito PDF: "ANALISAR OPERADOR OU PARÂMETRO DO NOVO ALGORITMO"
# Teste de Evaporação (Rho)
rodar_bateria("Fase 6: Calibração ACO (Evaporação Rho)", "6_calibracao_aco_rho", [
    {"nome": "ACO_Rho_0.5", "algoritmo": "ACO", "params": {"rho": 0.5, "num_formigas": 50}},
    {"nome": "ACO_Rho_0.1", "algoritmo": "ACO", "params": {"rho": 0.1, "num_formigas": 50}}
])

# --- ANÁLISE 7: CALIBRAGEM ACO (Parâmetro 2) ---
# Requisito PDF: "ANALISAR OPERADOR OU PARÂMETRO DO NOVO ALGORITMO"
# Teste de Tamanho da Colônia (Nº Formigas)
rodar_bateria("Fase 7: Calibração ACO (Nº Formigas)", "7_calibracao_aco_formigas", [
    {"nome": "ACO_20_Formigas", "algoritmo": "ACO", "params": {"num_formigas": 20, "rho": 0.1}},
    {"nome": "ACO_50_Formigas", "algoritmo": "ACO", "params": {"num_formigas": 50, "rho": 0.1}}
])

print("\n🎉 EXECUÇÃO TOTAL CONCLUÍDA! Verifique os novos gráficos em 'resultados/'.")